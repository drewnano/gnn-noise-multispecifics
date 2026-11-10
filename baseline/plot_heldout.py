"""Plot of the leave-one-topology-out experiment (baseline/heldout_topology.py).

    python -m baseline.plot_heldout       # writes baseline/results/heldout/heldout.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from baseline.train_baseline import node_coefficients

DIR = 'baseline/results/heldout'
MODELS = (('gin', 'GIN'), ('mlp_topo', 'MLP + topology'), ('mlp', 'MLP (blind)'))
BLUES = ['#5598e7', '#1c5cab', '#0d366b']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    runs = [json.loads(line) for line in open(f'{DIR}/runs.jsonl')]
    ref = json.load(open(f'{DIR}/reference.json'))
    C = node_coefficients('trispecific')
    folds = sorted({r['held'] for r in runs})

    def v(h, m, key='rmse'):
        return np.array([r[key] for r in runs if r['held'] == h and r['model'] == m])

    novelty = {h: min(float((C[h] - C[o]).norm()) for o in folds if o != h) for h in folds}
    order = sorted(folds, key=lambda h: ref['folds'][str(h)]['rmse_trainmean'])

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE,
                               gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.3))

    # (a) per fold: does the GIN beat knowing nothing about topology?
    yy = np.arange(len(order))[::-1]
    for y, h in zip(yy, order):
        base, orc = ref['folds'][str(h)]['rmse_trainmean'], ref['folds'][str(h)]['rmse_oracle']
        g, mt = v(h, 'gin'), v(h, 'mlp_topo')
        a.plot([orc, max(base, g.mean())], [y, y], color=GRID, linewidth=3, zorder=1)
        a.scatter(orc, y, s=46, marker='|', color=MUTED, linewidths=2, zorder=3,
                  label='oracle: this topology\'s own mean' if h == order[0] else None)
        a.scatter(base, y, s=70, marker='D', color=SURFACE, edgecolors=MUTED, linewidths=1.6, zorder=3,
                  label='baseline: training-set mean' if h == order[0] else None)
        a.scatter(mt.mean(), y, s=52, color=BLUES[2], alpha=0.5, zorder=3,
                  label='MLP + topology' if h == order[0] else None)
        a.scatter(g.mean(), y, s=86, color=BLUES[0], edgecolors=INK, linewidths=0.8, zorder=4,
                  label='GIN' if h == order[0] else None)
    a.set_yticks(yy)
    a.set_yticklabels([f'{h}   (novelty {novelty[h]:.1f})' for h in order], fontsize=8.5, color=INK)
    a.set_ylabel('Held-out topology', color=INK, fontsize=9)
    a.set_xlabel('Test RMSE on the held-out topology', color=INK, fontsize=9)
    a.set_title('a  The GIN beats "know nothing about topology" in 2 of 9 folds', loc='left', fontsize=10.5, color=INK)
    a.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    style(a)
    a.text(0.985, 0.60, 'right of the diamond =\nworse than ignoring topology', transform=a.transAxes,
           ha='right', fontsize=8, color=MUTED)

    # (b) the same trained models, in-distribution vs held-out
    xx = np.arange(len(MODELS))
    for i, (m, name) in enumerate(MODELS):
        seen = np.mean([v(h, m, 'rmse_seen').mean() for h in folds])
        held = np.mean([v(h, m).mean() for h in folds])
        b.plot([i - 0.17, i + 0.17], [seen, held], color=GRID, linewidth=3, zorder=1)
        b.scatter(i - 0.17, seen, s=95, color=BLUES[1], zorder=3, label='seen topologies' if i == 0 else None)
        b.scatter(i + 0.17, held, s=95, color=SURFACE, edgecolors=BLUES[1], linewidths=2, zorder=3,
                  label='held-out topology' if i == 0 else None)
        b.annotate(f'{seen:.1f}', (i - 0.17, seen), textcoords='offset points', xytext=(0, -15),
                   ha='center', fontsize=8, color=INK)
        b.annotate(f'{held:.1f}', (i + 0.17, held), textcoords='offset points', xytext=(0, 9),
                   ha='center', fontsize=8, color=INK)
    base = np.mean([ref['folds'][str(h)]['rmse_trainmean'] for h in folds])
    b.axhline(base, color=MUTED, linewidth=1, linestyle='--')
    b.text(-0.45, base + 0.9, 'training-set mean', fontsize=8, color=MUTED, ha='left')
    b.set_xticks(xx)
    b.set_xticklabels([n for _, n in MODELS], fontsize=8.5, color=INK)
    b.set_xlim(-0.5, len(MODELS) - 0.5)
    b.set_ylim(0, 36)
    b.set_ylabel('Test RMSE, mean over 9 folds', color=INK, fontsize=9)
    b.set_title('b  Same trained models, in and out of distribution', loc='left', fontsize=10.5, color=INK)
    b.legend(frameon=False, fontsize=8, loc='upper left')
    style(b)
    b.grid(axis='x', visible=False)
    b.grid(axis='y', color=GRID, linewidth=0.8)

    fig.text(0.01, 0.01, 'Trispecific, clean labels; pool 5,000 molecules minus the held-out topology; mean of 5 '
             'seeds. Novelty = distance from the held-out topology\'s node-coefficient vector to the nearest one in '
             'training. Spearman(novelty, GIN excess over baseline) = 0.00.', fontsize=7.5, color=MUTED)
    fig.savefig(f'{DIR}/heldout.png', dpi=160, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {DIR}/heldout.png')


if __name__ == '__main__':
    main()
