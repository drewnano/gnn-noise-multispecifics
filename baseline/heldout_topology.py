"""Leave-one-topology-out: can the GIN predict a format it has never seen?

Run from repo root (detached, so a closed session doesn't kill it):
    nohup python -m baseline.heldout_topology > baseline/results/heldout/run.log 2>&1 &
    python -m baseline.heldout_topology --summarize     # rebuild tables from runs.jsonl at any point

This targets the one claim a per-format lookup table cannot satisfy. The paper (Sec. 3.1) says the
format-conditioned MLP "is restricted to the fixed set of architectures defined in the training set,
whereas the GNN can learn generalized topological rules that could theoretically extend to novel, unseen
formats." Within the training set that claim is untestable: y = c_A.f is linear in f with c_A fixed per
format (see node_coefficients), every Fab sits at a fixed node index, so domain identity is readable from
a node's own features and knowing the format label is sufficient.

Each of the 9 trispecific topologies is held out in turn. Training molecules come only from the other 8;
the model is then scored on the held-out topology (`rmse`) and, for context, on the 8 seen ones
(`rmse_seen`). The MLP's embedding row for the held-out format never receives a gradient, so it stays at
its initialization -- its failure is by construction and is not the interesting part. The question is
whether the GIN beats the two topology-free baselines stored in reference.json:
    rmse_trainmean  predicting the mean of the training labels
    rmse_oracle     predicting the held-out topology's own true mean (the floor for any format-level model)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from baseline.train_baseline import TEST_SEED, DATA_CFG, make_data, make_task, split_train_val, train_one
from constants._connectivity import CONNECTIVITY_TYPES

FORMAT = 'trispecific'
MODELS = ('gin', 'mlp', 'mlp_topo')
TRAIN_ARGS = argparse.Namespace(format=FORMAT, dim=64, num_layers=3, multiplier=2, encoder='mean', conn_emb=False,
                                layernorm=False, batch_size=16, lr=1e-3, max_epochs=50, patience=50)
OUT = 'baseline/results/heldout'


def by_topology(data, keep, invert=False):
    return [d for d in data if (int(d.connectivity_type) in keep) != invert]


def run_job(job):
    torch.set_num_threads(1)
    held, seed, pool, test_held, test_seen = job
    train, val = split_train_val(pool, seed)
    t0 = time.time()
    out = []
    for kind in MODELS:
        r = train_one(kind, train, val, test_held, seed, TRAIN_ARGS, extra_tests={'seen': test_seen})
        r.update(held=held, seed=seed)
        out.append(r)
    print(f'  held={held} seed={seed}  ' + '  '.join(f'{r["model"]}={r["rmse"]:.2f}/{r["rmse_seen"]:.2f}' for r in out)
          + f'  [{time.time() - t0:.0f}s]', flush=True)
    return out


# ----------------------------------------------------------------------------- reporting

def summarize():
    runs = [json.loads(line) for line in open(f'{OUT}/runs.jsonl')]
    ref = json.load(open(f'{OUT}/reference.json'))
    folds = sorted({r['held'] for r in runs})

    def get(held, model, key):
        return np.array([r[key] for r in runs if r['held'] == held and r['model'] == model])

    def cell(held, model, key):
        v = get(held, model, key)
        return f'{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}' if len(v) else '—'

    lines = ['# Leave-one-topology-out (trispecific)', '',
             f'Pool {ref["pool_size"]} molecules minus the held-out topology, {ref["seeds"]} seeds, '
             f'{ref["test_size"]}-molecule test set. Clean labels. sd(y) over the whole test set = '
             f'{ref["sd_y"]:.2f}.', '',
             'The MLP + topology embedding row for a held-out format is never trained, so its failure there is '
             'by construction. The question is whether the GIN beats `train mean` (no topology knowledge) and how '
             'close it gets to `oracle` (that topology\'s own true mean).', '',
             '## Held-out topology: test RMSE', '',
             '| held out | n test | sd(y) in fold | train mean | oracle | GIN | MLP (blind) | MLP + topology |',
             '|---|---|---|---|---|---|---|---|']
    for h in folds:
        f = ref['folds'][str(h)]
        lines.append(f'| {h} | {f["n_test"]} | {f["sd_y"]:.2f} | {f["rmse_trainmean"]:.2f} | {f["rmse_oracle"]:.2f} | '
                     + ' | '.join(cell(h, m, 'rmse') for m in MODELS) + ' |')
    agg = {m: np.mean([get(h, m, 'rmse').mean() for h in folds]) for m in MODELS}
    lines += ['', f'Mean over folds — GIN {agg["gin"]:.2f}, blind MLP {agg["mlp"]:.2f}, '
                  f'MLP+topology {agg["mlp_topo"]:.2f}; train mean '
                  f'{np.mean([ref["folds"][str(h)]["rmse_trainmean"] for h in folds]):.2f}, oracle '
                  f'{np.mean([ref["folds"][str(h)]["rmse_oracle"] for h in folds]):.2f}.', '',
              '## Held-out topology: Spearman', '',
              '`nan` means the model predicted a constant on that fold, so rank correlation is undefined. That is the '
              'blind MLP\'s normal state here: it sits at predict-the-mean.', '',
              '| held out | GIN | MLP (blind) | MLP + topology |', '|---|---|---|---|']
    for h in folds:
        lines.append(f'| {h} | ' + ' | '.join(cell(h, m, 'spearman') for m in MODELS) + ' |')
    lines += ['', '## Seen topologies (same trained model): test RMSE', '',
              '| held out | GIN | MLP (blind) | MLP + topology |', '|---|---|---|---|']
    for h in folds:
        lines.append(f'| {h} | ' + ' | '.join(cell(h, m, 'rmse_seen') for m in MODELS) + ' |')
    lines.append('')
    with open(f'{OUT}/SUMMARY.md', 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {OUT}/SUMMARY.md from {len(runs)} model runs')


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument('--pool-size', type=int, default=5000, help='molecules drawn before dropping the held-out topology')
    p.add_argument('--test-size', type=int, default=9000, help='~1/9 of these land in each held-out fold')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--summarize', action='store_true', help='only rebuild SUMMARY.md from runs.jsonl')
    p.add_argument('--out', default=OUT)
    args = p.parse_args()
    OUT = args.out
    if args.summarize:
        return summarize()

    os.makedirs(OUT, exist_ok=True)
    DATA_CFG['motif_length'] = 3  # paper Table 1
    all_topo = list(range(len(CONNECTIVITY_TYPES[FORMAT])))
    t0 = time.time()
    fns = make_task(FORMAT)
    test = make_data(fns, FORMAT, args.test_size, TEST_SEED)
    pools = {s: make_data(fns, FORMAT, args.pool_size, s) for s in args.seeds}

    # Topology-free baselines per fold: both are properties of the data, not of any model.
    ref = dict(pool_size=args.pool_size, test_size=args.test_size, seeds=len(args.seeds),
               sd_y=float(torch.cat([d.y for d in test]).std()), folds={})
    for h in all_topo:
        y_h = torch.cat([d.y for d in by_topology(test, {h})])
        y_tr = torch.cat([d.y for d in by_topology(pools[args.seeds[0]], {h}, invert=True)])
        ref['folds'][str(h)] = dict(n_test=len(y_h), sd_y=float(y_h.std()),
                                    rmse_trainmean=float((y_h - y_tr.mean()).pow(2).mean().sqrt()),
                                    rmse_oracle=float((y_h - y_h.mean()).pow(2).mean().sqrt()))
    with open(f'{OUT}/reference.json', 'w') as f:
        json.dump(dict(ref, train_args=vars(TRAIN_ARGS), data_cfg=DATA_CFG), f, indent=2)

    done = set()
    if os.path.exists(f'{OUT}/runs.jsonl'):  # resume: skip (held, seed, model) already complete
        for line in open(f'{OUT}/runs.jsonl'):
            r = json.loads(line)
            done.add((r['held'], r['seed'], r['model']))
    jobs = []
    for h in all_topo:
        for s in args.seeds:
            if not all((h, s, m) in done for m in MODELS):
                # The seen-topology set is context only, so subsample it: it ships to every worker.
                jobs.append((h, s, by_topology(pools[s], {h}, invert=True),
                             by_topology(test, {h}), by_topology(test, {h}, invert=True)[:2000]))
    print(f'data generated in {time.time() - t0:.0f}s; {len(jobs)} jobs on {args.workers} workers', flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(f'{OUT}/runs.jsonl', 'a') as f:
        for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
            for r in fut.result():
                f.write(json.dumps(r) + '\n')
            f.flush()
    summarize()


if __name__ == '__main__':
    main()
