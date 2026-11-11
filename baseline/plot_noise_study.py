"""Plot of the assay-noise study (baseline/noise_study.py).

    python -m baseline.plot_noise_study      # writes baseline/results/noise_study/noise_study.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np

DIR = 'baseline/results/noise_study'
SIZES = [100, 200, 500, 1000, 2000, 5000]
FAMILIES = (('Fold error (unbiased)', ['clean', 'lognormal_0.2', 'lognormal_0.35'], ['none', '1.5-fold', '2-fold']),
            ('Batch effects confounded with format', ['clean', 'batch_rho0', 'batch_rho0.5', 'batch_rho1'],
             ['none', 'ρ = 0', 'ρ = 0.5', 'ρ = 1']),
            ('Saturating readout', ['clean', 'sat_1', 'sat_0.5', 'sat_0.25'], ['none', '1 sd', '0.5 sd', '0.25 sd']))
LABELS = {'clean': 'clean', 'lognormal_0.2': '1.5-fold error', 'lognormal_0.35': '2-fold error',
          'batch_rho0': 'batch, ρ = 0', 'batch_rho0.5': 'batch, ρ = 0.5', 'batch_rho1': 'batch, ρ = 1',
          'sat_1': 'saturation 1 sd', 'sat_0.5': 'saturation 0.5 sd', 'sat_0.25': 'saturation 0.25 sd'}
# Same ramp as plot_fig2.py: darker = more severe; clean in neutral ink
BLUES = ['#5598e7', '#1c5cab', '#0d366b']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax, log_x=True):
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(SIZES)
        ax.set_xticklabels([f'{s:,}' for s in SIZES], fontsize=8)
        ax.minorticks_off()
    ax.grid(axis='y' if log_x else 'x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    runs = [json.loads(line) for line in open(f'{DIR}/runs.jsonl')]
    ref = json.load(open(f'{DIR}/reference.json'))

    def vals(cond, size, model, key='rmse'):
        return np.array([r[key] for r in runs if r['condition'] == cond and r['size'] == size and r['model'] == model])

    fig = plt.figure(figsize=(16, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.42, 1.2], wspace=0.28, hspace=0.42)  # col 3 = spacer for g's labels

    # (a-f) RMSE against the true labels: rows = model, columns = noise family
    for col, (family, conds, levels) in enumerate(FAMILIES):
        for row, (model, name) in enumerate((('gin', 'GIN'), ('mlp_topo', 'MLP + topology'))):
            ax = fig.add_subplot(gs[row, col])
            for i, (cond, level) in enumerate(zip(conds, levels)):
                color = MUTED if cond == 'clean' else BLUES[i - 1 + (3 - (len(conds) - 1))]
                means = [vals(cond, s, model).mean() for s in SIZES]
                ax.plot(SIZES, means, color=color, linewidth=2, marker='o', markersize=4,
                        linestyle=':' if cond == 'clean' else '-', label=level)
            ax.axhline(ref['sd_y'], color=MUTED, linewidth=0.8, linestyle='--')
            ax.axhline(ref['topo_only'], color=MUTED, linewidth=0.8, linestyle='--')
            ax.set_ylim(4, 31)
            style(ax)
            ax.set_title(f'{"abc"[col] if row == 0 else "def"[col]}  {name}', loc='left', fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel('Test RMSE vs true labels', color=INK, fontsize=9)
                ax.text(SIZES[0], ref['sd_y'] + 0.5, 'predict the mean', fontsize=7.5, color=MUTED)
                ax.text(SIZES[0], ref['topo_only'] - 1.6, 'topology alone', fontsize=7.5, color=MUTED)
            if row == 0:
                ax.annotate(family, (0.5, 1.2), xycoords='axes fraction', ha='center', fontsize=11,
                            color=INK, weight='bold')
                ax.legend(frameon=False, fontsize=7.5, loc='upper center', ncol=2, title='severity', title_fontsize=7.5)
            else:
                ax.set_xlabel('Training molecules (train + val)', color=INK, fontsize=9)

    # (g) what a random split reports vs the truth, N = 5000: improvement over the blind MLP
    ax = fig.add_subplot(gs[:, 4])
    conds = [c for _, cs, _ in FAMILIES for c in cs if c != 'clean']
    conds = ['clean'] + conds
    yy = np.arange(len(conds))[::-1]
    for y, cond in zip(yy, conds):
        blind_obs, blind_true = vals(cond, 5000, 'mlp', 'rmse_obs'), vals(cond, 5000, 'mlp')
        obs = 1 - vals(cond, 5000, 'mlp_topo', 'rmse_obs').mean() / blind_obs.mean()
        true = 1 - vals(cond, 5000, 'mlp_topo').mean() / blind_true.mean()
        ax.plot([true, obs], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter(obs, y, s=60, color=SURFACE, edgecolors=BLUES[1], linewidths=1.8, zorder=3,
                   label='reported (noisy test labels)' if cond == 'clean' else None)
        ax.scatter(true, y, s=60, color=BLUES[1], zorder=3, label='real (true test labels)' if cond == 'clean' else None)
    ax.set_yticks(yy)
    ax.set_yticklabels([LABELS[c] for c in conds], fontsize=8.5, color=INK)
    ax.set_xlim(-0.05, 0.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    style(ax, log_x=False)
    ax.set_xlabel('RMSE reduction of MLP + topology over blind MLP', color=INK, fontsize=9)
    ax.set_title('g  Random-split evaluation vs truth (N = 5,000)', loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    fig.text(0.01, 0.01, 'Trispecific; mean-pooled residue embeddings; 55.3k params per model; 50 epochs; '
             'mean of 5 seeds; 2,000-molecule test set. All noisy conditions except "clean" also carry 1.5-fold error. '
             'Blind MLP stays at predict-the-mean in every condition.', fontsize=7.5, color=MUTED)
    fig.savefig(f'{DIR}/noise_study.png', dpi=160, facecolor=SURFACE, bbox_inches='tight')
    print(f'wrote {DIR}/noise_study.png')


if __name__ == '__main__':
    main()
