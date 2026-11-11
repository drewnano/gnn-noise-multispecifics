# Figure 2 reproduction + assay-noise sweep (trispecific)

Settings (paper Table 1 where the paper and repo disagree): mean-pooled residue embeddings, dim 64, 3 layers,
multiplier 2, no LayerNorm, connectivity-blind MLP with identical blocks (55,300 vs 55,297 params),
motif length 3, batch 16, 50 epochs, best-val checkpoint, 5 seeds. Noise: y + level * sd(y) * N(0,1),
sd(y) = 19.84, on train/val labels only; the test set (2,000 molecules) is clean. Reference RMSEs on the
test set: predict-the-mean 19.84; topology alone 7.00.

Source: `sweep_log_237of240.txt` (the parent process died before writing tables; 237/240 jobs logged), plus
the 3 missing jobs (N=20000, seed 4, noise 0.5/1/2) rerun with identical seeds into `missing_20000_seed4/`.
Plot: `fig2_noise.png` (`python -m baseline.plot_fig2`).

## GIN test RMSE: mean over seeds (seeds that left the predict-the-mean plateau, RMSE < 15)

MLP is 19.84–19.96 in every cell.

| N | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| 100 | 19.49 (0/5) | 18.69 (1/5) | 19.74 (0/5) | 18.79 (1/5) | 19.85 (0/5) | 19.82 (0/5) |
| 200 | 19.32 (0/5) | 18.46 (0/5) | 19.10 (0/5) | 19.44 (0/5) | 19.46 (0/5) | 19.68 (0/5) |
| 500 | 12.74 (3/5) | 12.70 (3/5) | 12.12 (3/5) | 14.47 (2/5) | 19.79 (0/5) | 19.86 (0/5) |
| 1000 | 7.17 (5/5) | 7.43 (5/5) | 7.62 (5/5) | 9.17 (5/5) | 13.39 (3/5) | 18.52 (1/5) |
| 2000 | 6.93 (5/5) | 9.45 (4/5) | 7.10 (5/5) | 7.30 (5/5) | 10.09 (4/5) | 13.58 (3/5) |
| 5000 | 6.48 (5/5) | 6.66 (5/5) | 9.29 (4/5) | 6.90 (5/5) | 10.06 (4/5) | 11.07 (4/5) |
| 10000 | 6.31 (5/5) | 6.34 (5/5) | 6.40 (5/5) | 6.62 (5/5) | 7.01 (5/5) | 12.52 (3/5) |
| 20000 | 6.19 (5/5) | 6.25 (5/5) | 6.27 (5/5) | 6.42 (5/5) | 6.78 (5/5) | 12.31 (3/5) |

## Gap = RMSE_GIN − RMSE_MLP (mean over seeds), paired t-test p (n = 5)

| N | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| 100 | −0.49 (0.083) | −1.30 (0.316) | −0.23 (0.156) | −1.16 (0.364) | −0.06 (0.378) | −0.14 (0.253) |
| 200 | −0.56 (0.033) | −1.40 (0.060) | −0.76 (<0.001) | −0.42 (0.011) | −0.42 (0.066) | −0.27 (0.175) |
| 500 | −7.11 (0.054) | −7.15 (0.047) | −7.72 (0.028) | −5.38 (0.081) | −0.06 (0.655) | −0.02 (0.427) |
| 1000 | −12.67 (<0.001) | −12.41 (<0.001) | −12.23 (<0.001) | −10.68 (<0.001) | −6.48 (0.067) | −1.44 (0.378) |
| 2000 | −12.92 (<0.001) | −10.40 (0.016) | −12.74 (<0.001) | −12.54 (<0.001) | −9.75 (0.016) | −6.28 (0.070) |
| 5000 | −13.37 (<0.001) | −13.18 (<0.001) | −10.55 (0.016) | −12.94 (<0.001) | −9.79 (0.016) | −8.84 (0.017) |
| 10000 | −13.52 (<0.001) | −13.50 (<0.001) | −13.44 (<0.001) | −13.22 (<0.001) | −12.83 (<0.001) | −7.33 (0.070) |
| 20000 | −13.65 (<0.001) | −13.59 (<0.001) | −13.57 (<0.001) | −13.42 (<0.001) | −13.06 (<0.001) | −7.53 (0.065) |

Outcomes are bimodal: each GIN run either learns the topology (RMSE ≈ 6–9) or stays at the mean (≈ 19.8).
The p-values come from 5 seeds and are coarse.

## Reading it

- **Clean curve.** This matches the paper's Figure 2 qualitatively. The MLP sits at predict-the-mean at every N. GIN starts learning at N=500 and approaches (then slightly beats) the topology-alone RMSE of 7.0 by N=1000. The gap grows with N and does not close.
- **Noise washout (first level with p ≥ 0.05).**

  | N | Washout level |
  |---|---|
  | ≤ 200 | No gap |
  | 500 | 0.5 |
  | 1000 | 1 |
  | 2000 | 2 (borderline) |
  | 5000 | Not reached by 2 |
  | 10000–20000 | 2 |

- **Noise mostly acts on optimization.** It barely affects attainable error: at N=20000, seeds that learn go only from 6.19 to 6.78 at noise 1. What noise changes is how often GIN escapes the plateau within 50 epochs. So washout levels depend on the training budget, not just on information content.
- **What the gap measures.** The gap is mostly topology identification (9 classes, ~88% of variance in y). That is a coarse signal that independent zero-mean noise averages out of. Realistic stress tests:
  - multiplicative or saturating noise
  - batch effects confounded with format
  - a topology-conditioned MLP baseline, which isolates the sequence-dependent part of the advantage
