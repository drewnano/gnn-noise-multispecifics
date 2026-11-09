"""Baseline: GIN vs parameter-matched, connectivity-blind MLP on Synapse (no added label noise).

Run from repo root (the dataset loads OAS pickles by relative path):
    python -m baseline.train_baseline --sizes 500 5000 --seeds 0 1 2 3 4

Design
  - Task (Ehrlich motifs + placements) is fixed by TASK_SEED for every run.
  - A seed s controls: which molecules are sampled for the train/val pool, the train/val split,
    model init, and batch order. GIN and MLP for the same (size, seed) see identical data.
  - Test set is one fixed held-out set of --test-size molecules (seed TEST_SEED), shared by all runs,
    so across-seed sd reflects training variability rather than test-set resampling.
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


def topology_blind_floor(test, fmt):
    """RMSE of the best connectivity-blind predictor given the *true* per-node f: E_topology[c]·f."""
    n = len(CONNECTIVITY_WEIGHTS[fmt][0])
    C = []
    for ei, w in zip(CONNECTIVITY_TYPES[fmt], CONNECTIVITY_WEIGHTS[fmt]):
        A = to_dense_adj(ei, max_num_nodes=n)[0]
        o = torch.ones(n)
        C.append(w * (1 + A @ o + A @ A @ o) / n)
    c_bar = torch.stack(C).mean(0)
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


def node_mlp(dim):
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


class GIN(nn.Module):
    def __init__(self, vocab_size, dim, num_layers):
        super().__init__()
        self.encoder = NodeEncoder(vocab_size, dim)
        self.layers = nn.ModuleList([GINConv(node_mlp(dim), train_eps=True) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x, batch, edge_index):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index).relu()
        return self.head(global_mean_pool(h, batch)).squeeze(-1)


class BlindMLP(nn.Module):
    """Same encoder, per-node MLP stack, pooling and head as GIN, minus aggregation. No edge_index input."""

    def __init__(self, vocab_size, dim, num_layers):
        super().__init__()
        self.encoder = NodeEncoder(vocab_size, dim)
        self.layers = nn.ModuleList([node_mlp(dim) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, x, batch):
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h).relu()
        return self.head(global_mean_pool(h, batch)).squeeze(-1)


MODELS = {'gin': GIN, 'mlp': BlindMLP}


def build_model(kind, args):
    return MODELS[kind](len(aa_to_idx), args.dim, args.num_layers)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, batch):
    if isinstance(model, GIN):
        return model(batch.x, batch.batch, batch.edge_index)
    return model(batch.x, batch.batch)


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
    size, seed, train, val, test, args = job
    t0 = time.time()
    out = []
    for kind in ('gin', 'mlp'):
        r = train_one(kind, train, val, test, seed, args)
        r['size'] = size
        out.append(r)
    print(f'  size={size} seed={seed}  GIN rmse={out[0]["rmse"]:.3f} (ep {out[0]["best_epoch"]})  '
          f'MLP rmse={out[1]["rmse"]:.3f} (ep {out[1]["best_epoch"]})  [{time.time() - t0:.0f}s]', flush=True)
    return out


# ----------------------------------------------------------------------------- reporting

def summarize(results, sizes, y_test_sd, floor, topo_only):
    lines = ['| N (train+val) | model | params | test RMSE (mean ± sd) | Spearman (mean ± sd) |',
             '|---|---|---|---|---|']
    gaps = []
    for size in sizes:
        r = {k: [x for x in results if x['size'] == size and x['model'] == k] for k in MODELS}
        for k in MODELS:
            rm = np.array([x['rmse'] for x in r[k]])
            sp = np.array([x['spearman'] for x in r[k]])
            lines.append(f'| {size} | {k.upper()} | {r[k][0]["params"]:,} | {rm.mean():.3f} ± {rm.std(ddof=1):.3f} '
                         f'| {sp.mean():.3f} ± {sp.std(ddof=1):.3f} |')
        g = {x['seed']: x['rmse'] for x in r['gin']}
        m = {x['seed']: x['rmse'] for x in r['mlp']}
        diffs = np.array([g[s] - m[s] for s in sorted(g)])
        sd_g, sd_m = np.std(list(g.values()), ddof=1), np.std(list(m.values()), ddof=1)
        gap = diffs.mean()
        gaps.append(dict(size=size, gap=gap, sd_gin=sd_g, sd_mlp=sd_m,
                         sd_pooled=float(np.sqrt((sd_g ** 2 + sd_m ** 2) / 2)),
                         sd_paired=diffs.std(ddof=1), n=len(diffs), diffs=diffs.tolist()))

    lines += ['', '| N | gap = RMSE_GIN − RMSE_MLP | sd GIN | sd MLP | |gap| / sd GIN | |gap| / sd MLP '
                  '| |gap| / pooled sd | sd of paired diffs | paired t |', '|---|---|---|---|---|---|---|---|---|']
    for g in gaps:
        t = g['gap'] / (g['sd_paired'] / np.sqrt(g['n']))
        lines.append(f'| {g["size"]} | {g["gap"]:+.3f} | {g["sd_gin"]:.3f} | {g["sd_mlp"]:.3f} '
                     f'| {abs(g["gap"]) / g["sd_gin"]:.1f} | {abs(g["gap"]) / g["sd_mlp"]:.1f} '
                     f'| {abs(g["gap"]) / g["sd_pooled"]:.1f} | {g["sd_paired"]:.3f} | {t:.1f} |')
    lines += ['', f'Reference points on the fixed test set: sd(y_test) = {y_test_sd:.3f} (RMSE of predicting the mean); '
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
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max-epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--out', default='baseline/results')
    args = p.parse_args()

    gin, mlp = build_model('gin', args), build_model('mlp', args)
    pg, pm = num_params(gin), num_params(mlp)
    print(f'params: GIN {pg:,}  MLP {pm:,}  relative diff {abs(pg - pm) / pg:.4%} '
          f'(GIN extra = {pg - pm} learnable eps scalars)')
    assert abs(pg - pm) / pg < 0.05, 'parameter counts differ by more than 5%'

    t0 = time.time()
    fns = make_task(args.format)
    test = make_data(fns, args.format, args.test_size, TEST_SEED)
    jobs = []
    for size in args.sizes:
        for seed in args.seeds:
            train, val = split_train_val(make_data(fns, args.format, size, seed), seed)
            jobs.append((size, seed, train, val, test, args))
    print(f'data generated in {time.time() - t0:.0f}s; training {len(jobs)} jobs on {args.workers} workers', flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = [r for out in ex.map(run_job, jobs) for r in out]
    print(f'training done in {time.time() - t0:.0f}s')

    y_test = torch.cat([d.y for d in test])
    table, gaps = summarize(results, args.sizes, float(y_test.std()), topology_blind_floor(test, args.format),
                            topology_only_rmse(test))
    print('\n' + table)

    os.makedirs(args.out, exist_ok=True)
    tag = f'{args.format}_' + '-'.join(map(str, args.sizes))
    with open(os.path.join(args.out, f'{tag}.json'), 'w') as f:
        json.dump(dict(config=vars(args), data_cfg=DATA_CFG, task_seed=TASK_SEED, test_seed=TEST_SEED,
                       params=dict(gin=pg, mlp=pm), runs=results, gaps=gaps), f, indent=2)
    with open(os.path.join(args.out, f'{tag}.md'), 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()
