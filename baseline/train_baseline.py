"""Baseline: GIN vs parameter-matched MLP on Synapse, optionally with label noise on train/val.

Run from repo root (the dataset loads OAS pickles by relative path):
    python -m baseline.train_baseline --sizes 500 5000 --seeds 0 1 2 3 4
    python -m baseline.train_baseline --sizes 5000 --noise-type y --noise-levels 0 0.1 0.25 0.5 1

Design
  - Task (Ehrlich motifs + placements) is fixed by TASK_SEED for every run.
  - A seed s controls: which molecules are sampled for the train/val pool, the train/val split,
    model init, and batch order. GIN and MLP for the same (size, seed) see identical data.
  - Test set is one fixed held-out set of --test-size molecules (seed TEST_SEED), shared by all runs,
    so across-seed sd reflects training variability rather than test-set resampling.
  - By default both models get the paper's learned per-topology scalar (use_connectivity_embedding in
    configurations/module/{gin,mlp}.yaml), concatenated to every node. --no-conn-emb makes the MLP
    fully connectivity-blind.
  - Label noise (--noise-type) is added to train/val y only; test y stays clean, so test RMSE measures
    recovery of the true function. Levels are in units of the clean test set's sd of the noised quantity:
      y: y + level * sd(y) * N(0,1)                      (assay / measurement noise on the readout)
      f: y + c_topology · (level * sd(f_fab) * N(0,1))   (noise in each Fab's score, propagated through Φ)
"""
import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import to_dense_adj

from constants._amino_acids import aa_to_idx
from constants._connectivity import CONNECTIVITY_TYPES, CONNECTIVITY_WEIGHTS
from holo.test_functions.closed_form import Ehrlich
from synthetic_dataset import AntibodySyntheticDataset
from utils import seed_all

# Data settings from configurations/datamodule/default.yaml (the Hydra train.py defaults)
DATA_CFG = dict(num_states=297, num_motifs=10, motif_length=4, edit_prob=0.3, noise_level=0.0)
NUM_FABS = {'bispecific': 2, 'trispecific': 3, 'tetraspecific': 4, 'pentaspecific': 5}
TASK_SEED = 0x00B1
TEST_SEED = 10_000


# ----------------------------------------------------------------------------- data

def make_task(fmt):
    """Ehrlich functions; motif placement uses the global RNG, so seed before constructing."""
    seed_all(TASK_SEED)
    return [Ehrlich(num_states=DATA_CFG['num_states'], num_motifs=DATA_CFG['num_motifs'],
                    motif_length=DATA_CFG['motif_length'], dim=297, random_seed=TASK_SEED + i,
                    epistasis_factor=0.0)
            for i in range(NUM_FABS[fmt])]


def make_data(fns, fmt, n, seed):
    ds = AntibodySyntheticDataset(num_states=DATA_CFG['num_states'], num_motifs=DATA_CFG['num_motifs'],
                                  motif_length=DATA_CFG['motif_length'], initial_seed=TASK_SEED,
                                  num_samples=n, noise_level=DATA_CFG['noise_level'],
                                  edit_prob=DATA_CFG['edit_prob'], input_type=fmt,
                                  ehrlich_functions=fns, seed=seed)
    return list(ds)


def split_train_val(pool, seed, val_frac=0.1):
    perm = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, int(round(val_frac * len(pool))))
    return [pool[i] for i in perm[n_val:]], [pool[i] for i in perm[:n_val]]


def node_coefficients(fmt):
    """Per-topology node coefficients c with y = c·f (Φ is linear in f; see baseline/phi_check.py)."""
    n = len(CONNECTIVITY_WEIGHTS[fmt][0])
    C = []
    for ei, w in zip(CONNECTIVITY_TYPES[fmt], CONNECTIVITY_WEIGHTS[fmt]):
        A = to_dense_adj(ei, max_num_nodes=n)[0]
        o = torch.ones(n)
        C.append(w * (1 + A @ o + A @ A @ o) / n)
    return torch.stack(C)


def add_label_noise(data, kind, level, seed, fmt, ref):
    """Copies of data with noisy y (see module docstring). ref holds sd(y) and sd(f_fab) of the clean test set."""
    if kind == 'none' or level == 0:
        return data
    g = torch.Generator().manual_seed(seed)
    C, n_fab = node_coefficients(fmt), NUM_FABS[fmt]
    out = []
    for d in data:
        d = d.clone()
        if kind == 'y':
            d.y = d.y + level * ref['sd_y'] * torch.randn(1, generator=g)
        else:
            eps = torch.zeros_like(d.fn_values)
            eps[:n_fab] = level * ref['sd_f'] * torch.randn(n_fab, generator=g)
            d.y = d.y + (C[d.connectivity_type] @ eps).reshape(1)
        out.append(d)
    return out


def topology_blind_floor(test, fmt):
    """RMSE of the best connectivity-blind predictor given the *true* per-node f: E_topology[c]·f."""
    c_bar = node_coefficients(fmt).mean(0)
    y = torch.cat([d.y for d in test])
    y_hat = torch.stack([d.fn_values for d in test]) @ c_bar
    return float((y - y_hat).pow(2).mean().sqrt())


def topology_only_rmse(test):
    """RMSE of E[y | topology] (test-set group means): what knowing only the topology buys."""
    y = torch.cat([d.y for d in test])
    t = torch.tensor([d.connectivity_type for d in test])
    y_hat = torch.zeros_like(y)
    for k in t.unique():
        y_hat[t == k] = y[t == k].mean()
    return float((y - y_hat).pow(2).mean().sqrt())


# ----------------------------------------------------------------------------- models

class NodeEncoder(nn.Module):
    """Embed integer AA indices and mean-pool over sequence positions -> one vector per node."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.emb(x.long()).mean(dim=1)


def node_mlp(in_dim, dim, multiplier=1, layernorm=False):
    """Per-node update. multiplier=1: Linear-ReLU-Linear (first baseline). multiplier>1: the repo's GINConv MLP
    (models/layers/gnn.py), Linear(in, m*dim) - [LayerNorm] - GELU - Linear(m*dim, dim). LayerNorm is off by
    default: with mean-pooled encoders the Fab nodes look alike, topology reaches GIN mostly through the
    magnitude of the summed messages, and LayerNorm removes it (GIN escaped the predict-the-mean plateau in
    1/3 seeds with it vs 3/3 without, N=2000)."""
    if multiplier == 1:
        return nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    h = multiplier * dim
    return nn.Sequential(nn.Linear(in_dim, h), *([nn.LayerNorm(h)] if layernorm else []), nn.GELU(), nn.Linear(h, dim))


class ConvNodeEncoder(nn.Module):
    """Position-aware alternative: embed, two 1D convs over the sequence, max-pool -> one vector per node.
    Ehrlich scores reward motifs (with spacings) anywhere in the sequence, which mean-pooled embeddings
    (amino-acid composition only) largely cannot see. Lighter stand-in for the paper's transformer encoder."""

    def __init__(self, vocab_size, dim, kernel_size=9):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.convs = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2), nn.ReLU(),
                                   nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2), nn.ReLU())

    def forward(self, x):
        return self.convs(self.emb(x.long()).transpose(1, 2)).amax(dim=2)


ENCODERS = {'mean': NodeEncoder, 'cnn': ConvNodeEncoder}


class TopologyEncoder(nn.Module):
    """Node encoder plus, if num_topologies > 0, the paper's learned per-topology scalar on every node."""

    def __init__(self, vocab_size, dim, num_topologies, encoder='mean'):
        super().__init__()
        self.encoder = ENCODERS[encoder](vocab_size, dim)
        self.topo = nn.Embedding(num_topologies, 1) if num_topologies else None
        self.out_dim = dim + (1 if num_topologies else 0)

    def forward(self, x, batch, connectivity_type):
        h = self.encoder(x)
        if self.topo is not None:
            h = torch.cat([h, self.topo(connectivity_type.long())[batch]], dim=-1)
        return h


class GIN(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_topologies=0, encoder='mean', multiplier=1, layernorm=False):
        super().__init__()
        self.encoder = TopologyEncoder(vocab_size, dim, num_topologies, encoder)
        in_dims = [self.encoder.out_dim] + [dim] * (num_layers - 1)
        self.layers = nn.ModuleList([GINConv(node_mlp(i, dim, multiplier, layernorm), train_eps=True) for i in in_dims])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x, batch, connectivity_type, edge_index):
        h = self.encoder(x, batch, connectivity_type)
        for layer in self.layers:
            h = layer(h, edge_index).relu()
        return self.head(global_mean_pool(h, batch)).squeeze(-1)


class BlindMLP(nn.Module):
    """Same encoder, per-node MLP stack, pooling and head as GIN, minus aggregation. No edge_index input."""

    def __init__(self, vocab_size, dim, num_layers, num_topologies=0, encoder='mean', multiplier=1, layernorm=False):
        super().__init__()
        self.encoder = TopologyEncoder(vocab_size, dim, num_topologies, encoder)
        in_dims = [self.encoder.out_dim] + [dim] * (num_layers - 1)
        self.layers = nn.ModuleList([node_mlp(i, dim, multiplier, layernorm) for i in in_dims])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x, batch, connectivity_type):
        h = self.encoder(x, batch, connectivity_type)
        for layer in self.layers:
            h = layer(h).relu()
        return self.head(global_mean_pool(h, batch)).squeeze(-1)


MODELS = {'gin': GIN, 'mlp': BlindMLP}


def build_model(kind, args):
    num_topologies = len(CONNECTIVITY_TYPES[args.format]) if args.conn_emb else 0
    return MODELS[kind](len(aa_to_idx), args.dim, args.num_layers, num_topologies, args.encoder, args.multiplier,
                         args.layernorm)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, batch):
    if isinstance(model, GIN):
        return model(batch.x, batch.batch, batch.connectivity_type, batch.edge_index)
    return model(batch.x, batch.batch, batch.connectivity_type)


# ----------------------------------------------------------------------------- training

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    return torch.cat([predict(model, b) for b in loader])


def train_one(kind, train, val, test, seed, args):
    seed_all(seed)
    model = build_model(kind, args)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    y_tr = torch.cat([d.y for d in train])
    mu, sd = y_tr.mean(), y_tr.std()
    y_val = (torch.cat([d.y for d in val]) - mu) / sd

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val, batch_size=512)
    test_loader = DataLoader(test, batch_size=512)

    best_val, best_state, best_epoch, epoch = float('inf'), None, 0, 0
    for epoch in range(args.max_epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(predict(model, b), (b.y - mu) / sd)
            loss.backward()
            opt.step()
        val_mse = float(nn.functional.mse_loss(evaluate(model, val_loader), y_val))
        if val_mse < best_val:
            best_val, best_state, best_epoch = val_mse, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)

    y_te = torch.cat([d.y for d in test])
    pred = evaluate(model, test_loader) * sd + mu
    return dict(model=kind, seed=seed, n_train=len(train), n_val=len(val), params=num_params(model),
                rmse=float((pred - y_te).pow(2).mean().sqrt()),
                spearman=float(spearmanr(pred.numpy(), y_te.numpy()).statistic),
                best_epoch=best_epoch, epochs_run=epoch + 1, best_val_mse_std=best_val)


def run_job(job):
    torch.set_num_threads(1)
    size, level, seed, train, val, test, args = job
    t0 = time.time()
    out = []
    for kind in ('gin', 'mlp'):
        r = train_one(kind, train, val, test, seed, args)
        r.update(size=size, noise_level=level)
        out.append(r)
    print(f'  size={size} noise={level} seed={seed}  GIN rmse={out[0]["rmse"]:.3f} (ep {out[0]["best_epoch"]})  '
          f'MLP rmse={out[1]["rmse"]:.3f} (ep {out[1]["best_epoch"]})  [{time.time() - t0:.0f}s]', flush=True)
    return out


# ----------------------------------------------------------------------------- reporting

def summarize(results, cells, y_test_sd, floor, topo_only, noise_type):
    lines = [f'| N (train+val) | noise ({noise_type}) | model | params | test RMSE (mean ± sd) | Spearman (mean ± sd) |',
             '|---|---|---|---|---|---|']
    gaps = []
    for size, level in cells:
        r = {k: [x for x in results if x['size'] == size and x['noise_level'] == level and x['model'] == k]
             for k in MODELS}
        for k in MODELS:
            rm = np.array([x['rmse'] for x in r[k]])
            sp = np.array([x['spearman'] for x in r[k]])
            lines.append(f'| {size} | {level} | {k.upper()} | {r[k][0]["params"]:,} | {rm.mean():.3f} ± {rm.std(ddof=1):.3f} '
                         f'| {sp.mean():.3f} ± {sp.std(ddof=1):.3f} |')
        g = {x['seed']: x['rmse'] for x in r['gin']}
        m = {x['seed']: x['rmse'] for x in r['mlp']}
        diffs = np.array([g[s] - m[s] for s in sorted(g)])
        sd_g, sd_m = np.std(list(g.values()), ddof=1), np.std(list(m.values()), ddof=1)
        gaps.append(dict(size=size, noise_level=level, gap=diffs.mean(), sd_gin=sd_g, sd_mlp=sd_m,
                         sd_pooled=float(np.sqrt((sd_g ** 2 + sd_m ** 2) / 2)),
                         sd_paired=diffs.std(ddof=1), n=len(diffs), diffs=diffs.tolist()))

    lines += ['', '| N | noise | gap = RMSE_GIN − RMSE_MLP | sd GIN | sd MLP | |gap| / pooled sd '
                  '| sd of paired diffs | paired t |', '|---|---|---|---|---|---|---|---|']
    for g in gaps:
        t = g['gap'] / (g['sd_paired'] / np.sqrt(g['n']))
        lines.append(f'| {g["size"]} | {g["noise_level"]} | {g["gap"]:+.3f} | {g["sd_gin"]:.3f} | {g["sd_mlp"]:.3f} '
                     f'| {abs(g["gap"]) / g["sd_pooled"]:.1f} | {g["sd_paired"]:.3f} | {t:.1f} |')
    lines += ['', f'Reference points on the fixed (clean) test set: sd(y_test) = {y_test_sd:.3f} (RMSE of predicting the mean); '
                  f'best possible connectivity-blind predictor given the true per-node f (E_topology[c]·f) has '
                  f'RMSE = {floor:.3f}; topology alone, E[y | topology], has RMSE = {topo_only:.3f}; '
                  f'with topology and f both known, y is deterministic (floor 0).']
    return '\n'.join(lines), gaps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--format', default='trispecific')
    p.add_argument('--sizes', type=int, nargs='+', default=[500, 5000])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--test-size', type=int, default=2000)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=3)
    p.add_argument('--multiplier', type=int, default=1,
                   help='hidden width multiplier of the per-node MLPs; the paper (Table 1) uses 2')
    p.add_argument('--layernorm', action='store_true', help="LayerNorm in per-node MLPs (repo's GINConv; see node_mlp)")
    p.add_argument('--motif-length', type=int, default=DATA_CFG['motif_length'],
                   help='Ehrlich motif length; repo default.yaml uses 4, the paper (Table 1) 3')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--encoder', choices=list(ENCODERS), default='mean',
                   help='mean: mean-pooled embeddings (composition only); cnn: position-aware conv + max-pool')
    p.add_argument('--no-conn-emb', dest='conn_emb', action='store_false',
                   help="drop the paper's per-topology embedding (fully connectivity-blind MLP)")
    p.add_argument('--noise-type', choices=['none', 'y', 'f'], default='none')
    p.add_argument('--noise-levels', type=float, nargs='+', default=[0.0])
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--out', default='baseline/results')
    args = p.parse_args()
    DATA_CFG['motif_length'] = args.motif_length  # data is generated in this process only

    gin, mlp = build_model('gin', args), build_model('mlp', args)
    pg, pm = num_params(gin), num_params(mlp)
    print(f'params: GIN {pg:,}  MLP {pm:,}  relative diff {abs(pg - pm) / pg:.4%} '
          f'(GIN extra = {pg - pm} learnable eps scalars); topology embedding: {args.conn_emb}')
    assert abs(pg - pm) / pg < 0.05, 'parameter counts differ by more than 5%'

    t0 = time.time()
    fns = make_task(args.format)
    test = make_data(fns, args.format, args.test_size, TEST_SEED)
    n_fab = NUM_FABS[args.format]
    ref = dict(sd_y=float(torch.cat([d.y for d in test]).std()),
               sd_f=float(torch.stack([d.fn_values[:n_fab] for d in test]).std()))
    levels = args.noise_levels if args.noise_type != 'none' else [0.0]
    jobs, cells = [], [(size, level) for size in args.sizes for level in levels]
    for size in args.sizes:
        for seed in args.seeds:
            train, val = split_train_val(make_data(fns, args.format, size, seed), seed)
            for level in levels:
                # Same molecules and same noise draw at every level; only its scale changes
                ns = seed * 1000
                jobs.append((size, level, seed, add_label_noise(train, args.noise_type, level, ns, args.format, ref),
                             add_label_noise(val, args.noise_type, level, ns + 500, args.format, ref), test, args))
    print(f'data generated in {time.time() - t0:.0f}s; noise reference sd(y)={ref["sd_y"]:.3f} '
          f'sd(f_fab)={ref["sd_f"]:.3f}; training {len(jobs)} jobs on {args.workers} workers', flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [r for out in ex.map(run_job, jobs) for r in out]
    print(f'training done in {time.time() - t0:.0f}s')

    y_test = torch.cat([d.y for d in test])
    table, gaps = summarize(results, cells, float(y_test.std()), topology_blind_floor(test, args.format),
                            topology_only_rmse(test), args.noise_type)
    print('\n' + table)

    os.makedirs(args.out, exist_ok=True)
    tag = (f'{args.format}_' + '-'.join(map(str, args.sizes)) + f'_{args.encoder}' + ('_connemb' if args.conn_emb else '_blind')
           + (f'_noise-{args.noise_type}' if args.noise_type != 'none' else ''))
    with open(os.path.join(args.out, f'{tag}.json'), 'w') as f:
        json.dump(dict(config=vars(args), data_cfg=DATA_CFG, task_seed=TASK_SEED, test_seed=TEST_SEED,
                       noise_ref=ref, params=dict(gin=pg, mlp=pm), runs=results, gaps=gaps), f, indent=2)
    with open(os.path.join(args.out, f'{tag}.md'), 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()
