"""Figure 2-style plot of the GIN vs MLP size x noise sweep.

Reads per-job results from the sweep log (the sweep's parent process died before writing its JSON) plus
the JSON of the rerun of the missing jobs, then writes baseline/results/fig2/fig2_noise.png.
    python -m baseline.plot_fig2
"""
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

DIR = 'baseline/results/fig2'
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SD_Y, TOPO_ONLY = 19.839, 6.998  # clean test set: predict-the-mean RMSE, E[y | topology] RMSE
# Single-hue ordinal ramp (blue 250 -> 700), darker = more noise; MLP in neutral ink
BLUES = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load():
    """{(size, level, seed): (gin_rmse, mlp_rmse)}"""
    runs = {}
    for line in open(f'{DIR}/sweep_log_237of240.txt'):
        m = re.search(r'size=(\d+) noise=([\d.]+) seed=(\d+)\s+GIN rmse=([\d.]+).*MLP rmse=([\d.]+)', line)
        if m:
            runs[(int(m[1]), float(m[2]), int(m[3]))] = (float(m[4]), float(m[5]))
    for path in glob.glob(f'{DIR}/missing_*/*.json'):
        by_key = {}
        for r in json.load(open(path))['runs']:
            by_key.setdefault((r['size'], float(r['noise_level']), r['seed']), {})[r['model']] = r['rmse']
        for k, v in by_key.items():
            runs[k] = (v['gin'], v['mlp'])
    return runs


def cell(runs, size, level):
    v = np.array([runs[(size, level, s)] for s in range(5) if (size, level, s) in runs])
    return v[:, 0], v[:, 1]


def style(ax):
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f'{s:,}' for s in SIZES], fontsize=8)
    ax.minorticks_off()
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel('Training molecules (train + val)', color=INK)


def main():
    runs = load()
    missing = [(s, l, d) for s in SIZES for l in LEVELS for d in range(5) if (s, l, d) not in runs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor='#fcfcfb')

    # (a) clean curve, every seed shown: GIN outcomes are bimodal (learns topology or stays at the mean)
    jit = np.linspace(-0.06, 0.06, 5)
    for kind, color, i in (('GIN', BLUES[3], 0), ('MLP', MUTED, 1)):
        means = []
        for s in SIZES:
            y = cell(runs, s, 0.0)[i]
            means.append(y.mean())
            a.scatter(s * 10 ** jit[:len(y)], y, s=22, color=color, alpha=0.45, linewidths=0)
        a.plot(SIZES, means, color=color, linewidth=2, marker='o', markersize=5, label=f'{kind} (mean of 5 seeds)')
    a.axhline(TOPO_ONLY, color=MUTED, linewidth=1, linestyle=':')
    a.text(SIZES[0], TOPO_ONLY - 1.3, 'topology alone, E[y | topology]', fontsize=8, color=MUTED)
    a.text(1000, SD_Y + 0.6, 'predict the mean', fontsize=8, color=MUTED)
    a.set_ylim(4, 22)
    a.set_ylabel('Test RMSE (clean labels)', color=INK)
    a.set_title('a  No added noise: GIN vs connectivity-blind MLP', loc='left', fontsize=11, color=INK)
    a.legend(frameon=False, fontsize=8, loc='center right')
    style(a)

    # (b) gap per noise level; hollow marker = not significant (paired t, p >= 0.05)
    for level, color in zip(LEVELS, BLUES):
        gaps, sig = [], []
        for s in SIZES:
            g, m = cell(runs, s, level)
            gaps.append((g - m).mean())
            sig.append(ttest_rel(g, m).pvalue < 0.05)
        # Darkest two steps are close in lightness, so the top level also gets a dashed line
        b.plot(SIZES, gaps, color=color, linewidth=2, linestyle='--' if level == LEVELS[-1] else '-',
               label=f'{level:g} × sd(y)')
        for s, gap, ok in zip(SIZES, gaps, sig):
            b.scatter(s, gap, s=40, zorder=3, color=color if ok else '#fcfcfb', edgecolors=color, linewidths=1.5)
    b.axhline(0, color=MUTED, linewidth=1)
    b.set_ylabel('Gap = RMSE(GIN) − RMSE(MLP)', color=INK)
    b.set_title('b  Assay noise on training labels', loc='left', fontsize=11, color=INK)
    b.legend(title='Label noise sd', title_fontsize=8, frameon=False, fontsize=8, loc='lower left')
    b.text(0.99, 0.90, 'filled: p < 0.05 (paired t, 5 seeds)\nhollow: not significant', transform=b.transAxes,
           ha='right', va='top', fontsize=8, color=MUTED)
    style(b)

    note = 'Trispecific; mean-pooled residue embeddings; 55.3k params each; 50 epochs; 2,000-molecule clean test set.'
    if missing:
        note += f'  Missing runs: {missing}'
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f'{DIR}/fig2_noise.png', dpi=160, facecolor=fig.get_facecolor())
    print(f'wrote {DIR}/fig2_noise.png; missing runs: {missing or "none"}')


if __name__ == '__main__':
    main()
