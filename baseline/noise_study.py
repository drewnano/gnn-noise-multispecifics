"""Assay-noise study: noise that cannot simply be averaged out with more data.

Run from repo root (detached, so a closed session doesn't kill it):
    nohup python -m baseline.noise_study > baseline/results/noise_study/run.log 2>&1 &
    python -m baseline.noise_study --summarize        # rebuild tables from runs.jsonl at any point

Observation model, applied to every label the "lab" reports (train, val and the observed test labels):
    signal   = sat(y)                                   saturating readout (tanh around the median of y)
    measured = gain_k * signal * exp(log_sd * N(0,1)) + offset_k
  - sat: half-range w * sd(y); w = None means a linear readout. Compresses both ends (ceiling and floor).
  - campaign k: one of 9 campaigns with gain_k = exp(batch_gain * N(0,1)) and offset_k = batch_offset * sd(y) * N(0,1).
    With probability rho a molecule is measured in its own topology's campaign, otherwise in a random one,
    so rho=1 confounds campaign with format completely.
  - log-normal fold error: log_sd 0.2 ≈ 1.5-fold and 0.35 ≈ 2-fold at 95%.

Each run is scored against the clean test labels (truth) and against observed test labels from the same
campaigns (what a random split of a real dataset would report). Three models, matched except for their
input: GIN, connectivity-blind MLP, and an MLP given the per-topology embedding (mlp_topo).
Settings are the paper's (see baseline/PAPER_VS_REPO.md), as in the Figure 2 reproduction.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.stats import ttest_rel

from baseline.train_baseline import TEST_SEED, DATA_CFG, make_data, make_task, split_train_val, train_one
from constants._connectivity import CONNECTIVITY_TYPES

FORMAT = 'trispecific'
MODELS = ('gin', 'mlp', 'mlp_topo')
BATCH = dict(batch_gain=0.2, batch_offset=0.5)
CONDITIONS = {
    'clean':           dict(),
    'lognormal_0.2':   dict(log_sd=0.2),
    'lognormal_0.35':  dict(log_sd=0.35),
    'batch_rho0':      dict(log_sd=0.2, rho=0.0, **BATCH),
    'batch_rho0.5':    dict(log_sd=0.2, rho=0.5, **BATCH),
    'batch_rho1':      dict(log_sd=0.2, rho=1.0, **BATCH),
    'sat_1':           dict(log_sd=0.2, sat=1.0),
    'sat_0.5':         dict(log_sd=0.2, sat=0.5),
    'sat_0.25':        dict(log_sd=0.2, sat=0.25),
}
TRAIN_ARGS = argparse.Namespace(format=FORMAT, dim=64, num_layers=3, multiplier=2, encoder='mean', conn_emb=False,
                                layernorm=False, batch_size=16, lr=1e-3, max_epochs=50, patience=50)
OUT = 'baseline/results/noise_study'


def campaigns(cond, seed):
    """Per-campaign (gain, offset in sd(y) units); shared by train, val and observed-test labels of one run."""
    g = torch.Generator().manual_seed(seed * 7919 + 1)
    k = len(CONNECTIVITY_TYPES[FORMAT])
    gain = torch.exp(cond.get('batch_gain', 0.0) * torch.randn(k, generator=g))
    offset = cond.get('batch_offset', 0.0) * torch.randn(k, generator=g)
    return gain, offset


def observe(data, cond, camp, ref, seed):
    """Observed labels (tensor, same order as data) under the observation model in the module docstring."""
    g = torch.Generator().manual_seed(seed)
    y = torch.cat([d.y for d in data])
    topo = torch.tensor([d.connectivity_type for d in data])
    if cond.get('sat') is not None:
        half = cond['sat'] * ref['sd_y']
        y = ref['median_y'] + half * torch.tanh((y - ref['median_y']) / half)
    if 'rho' in cond:
        gain, offset = camp
        own = torch.rand(len(y), generator=g) < cond['rho']
        k = torch.where(own, topo, torch.randint(len(gain), (len(y),), generator=g))
        y = gain[k] * y + offset[k] * ref['sd_y']
    if cond.get('log_sd'):
        y = y * torch.exp(cond['log_sd'] * torch.randn(len(y), generator=g))
    return y


def with_labels(data, y):
    out = []
    for d, yi in zip(data, y):
        d = d.clone()
        d.y = yi.reshape(1)
        out.append(d)
    return out


def run_job(job):
    torch.set_num_threads(1)
    name, size, seed, pool, test, ref = job
    cond = CONDITIONS[name]
    camp = campaigns(cond, seed)
    pool = with_labels(pool, observe(pool, cond, camp, ref, seed * 1000 + 1))
    train, val = split_train_val(pool, seed)
    y_test_obs = observe(test, cond, camp, ref, seed * 1000 + 2)
    t0 = time.time()
    out = []
    for kind in MODELS:
        r = train_one(kind, train, val, test, seed, TRAIN_ARGS, extra_targets={'obs': y_test_obs})
        r.update(condition=name, size=size)
        out.append(r)
    print(f'  {name:15s} N={size:<5d} seed={seed}  ' + '  '.join(f'{r["model"]}={r["rmse"]:.2f}' for r in out)
          + f'  [{time.time() - t0:.0f}s]', flush=True)
    return out


# ----------------------------------------------------------------------------- reporting

def summarize():
    runs = [json.loads(line) for line in open(f'{OUT}/runs.jsonl')]
    ref = json.load(open(f'{OUT}/reference.json'))
    names = [c for c in CONDITIONS if any(r['condition'] == c for r in runs)]
    sizes = sorted({r['size'] for r in runs})

    def get(c, n, model, key):
        rs = sorted((r for r in runs if r['condition'] == c and r['size'] == n and r['model'] == model),
                    key=lambda r: r['seed'])
        return np.array([r[key] for r in rs]), [r['seed'] for r in rs]

    def gap(c, n, a, b, key):
        ya, sa = get(c, n, a, key)
        yb, sb = get(c, n, b, key)
        common = sorted(set(sa) & set(sb))
        ya = np.array([ya[sa.index(s)] for s in common])
        yb = np.array([yb[sb.index(s)] for s in common])
        if len(common) < 2:
            return '—'
        d = ya - yb
        return f'{d.mean():+.2f} (p={ttest_rel(ya, yb).pvalue:.3f})'

    lines = ['# Assay-noise study (trispecific)', '',
             f'Clean test set: sd(y) = {ref["sd_y"]:.2f}, topology alone RMSE = {ref["topo_only"]:.2f}. '
             'Mean over seeds; gaps are paired over seeds (p from paired t). "clean" = scored against true test labels, '
             '"obs" = scored against observed test labels from the same campaigns.', '',
             'Conditions: ' + '; '.join(f'`{c}` {CONDITIONS[c] or "no noise"}' for c in names), '']
    for key, title in (('rmse', 'Test RMSE vs clean labels'), ('rmse_obs', 'Test RMSE vs observed labels'),
                       ('spearman', 'Spearman vs clean labels'), ('spearman_obs', 'Spearman vs observed labels')):
        lines += [f'## {title}', '', '| condition | N | GIN | MLP (blind) | MLP + topology | GIN − MLP | GIN − MLP+topo |',
                  '|---|---|---|---|---|---|---|']
        for c in names:
            for n in sizes:
                m = [get(c, n, k, key)[0] for k in MODELS]
                if not all(len(x) for x in m):
                    continue
                cells = [f'{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}' for x in m]
                lines.append(f'| {c} | {n} | ' + ' | '.join(cells) + f' | {gap(c, n, "gin", "mlp", key)} '
                             f'| {gap(c, n, "gin", "mlp_topo", key)} |')
        lines.append('')
    with open(f'{OUT}/SUMMARY.md', 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {OUT}/SUMMARY.md from {len(runs)} model runs')


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+', default=[100, 200, 500, 1000, 2000, 5000])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument('--test-size', type=int, default=2000)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--summarize', action='store_true', help='only rebuild SUMMARY.md from runs.jsonl')
    p.add_argument('--out', default=OUT)
    args = p.parse_args()
    OUT = args.out
    if args.summarize:
        return summarize()

    os.makedirs(OUT, exist_ok=True)
    DATA_CFG['motif_length'] = 3  # paper Table 1
    t0 = time.time()
    fns = make_task(FORMAT)
    test = make_data(fns, FORMAT, args.test_size, TEST_SEED)
    y = torch.cat([d.y for d in test])
    topo = torch.tensor([d.connectivity_type for d in test])
    topo_mean = torch.zeros_like(y)
    for k in topo.unique():
        topo_mean[topo == k] = y[topo == k].mean()
    ref = dict(sd_y=float(y.std()), median_y=float(y.median()), topo_only=float((y - topo_mean).pow(2).mean().sqrt()))
    with open(f'{OUT}/reference.json', 'w') as f:
        json.dump(dict(ref, conditions=CONDITIONS, train_args=vars(TRAIN_ARGS), data_cfg=DATA_CFG), f, indent=2)

    done = set()
    if os.path.exists(f'{OUT}/runs.jsonl'):  # resume: skip (condition, size, seed) already complete
        for line in open(f'{OUT}/runs.jsonl'):
            r = json.loads(line)
            done.add((r['condition'], r['size'], r['seed'], r['model']))
    jobs = []
    for size in args.sizes:
        for seed in args.seeds:
            pool = make_data(fns, FORMAT, size, seed)
            for name in args.conditions:
                if not all((name, size, seed, m) in done for m in MODELS):
                    jobs.append((name, size, seed, pool, test, ref))
    print(f'data generated in {time.time() - t0:.0f}s; {len(jobs)} jobs on {args.workers} workers', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(f'{OUT}/runs.jsonl', 'a') as f:
        for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
            for r in fut.result():
                f.write(json.dumps(r) + '\n')
            f.flush()
    summarize()


if __name__ == '__main__':
    main()
