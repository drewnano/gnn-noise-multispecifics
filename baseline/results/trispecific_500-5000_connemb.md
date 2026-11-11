| N (train+val) | noise (none) | model | params | test RMSE (mean ± sd) | Spearman (mean ± sd) |
|---|---|---|---|---|---|
| 500 | 0.0 | GIN | 30,605 | 7.622 ± 0.267 | 0.916 ± 0.009 |
| 500 | 0.0 | MLP | 30,602 | 7.730 ± 0.181 | 0.915 ± 0.003 |
| 5000 | 0.0 | GIN | 30,605 | 6.557 ± 0.066 | 0.942 ± 0.002 |
| 5000 | 0.0 | MLP | 30,602 | 6.530 ± 0.073 | 0.943 ± 0.002 |

| N | noise | gap = RMSE_GIN − RMSE_MLP | sd GIN | sd MLP | |gap| / pooled sd | sd of paired diffs | paired t |
|---|---|---|---|---|---|---|---|
| 500 | 0.0 | -0.108 | 0.267 | 0.181 | 0.5 | 0.328 | -0.7 |
| 5000 | 0.0 | +0.027 | 0.066 | 0.073 | 0.4 | 0.060 | 1.0 |

Reference points on the fixed (clean) test set: sd(y_test) = 29.253 (RMSE of predicting the mean); best possible connectivity-blind predictor given the true per-node f (E_topology[c]·f) has RMSE = 28.531; topology alone, E[y | topology], has RMSE = 7.492; with topology and f both known, y is deterministic (floor 0).
