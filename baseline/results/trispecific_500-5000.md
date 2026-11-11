| N (train+val) | model | params | test RMSE (mean ± sd) | Spearman (mean ± sd) |
|---|---|---|---|---|
| 500 | GIN | 30,532 | 15.826 ± 10.602 | 0.686 ± 0.310 |
| 500 | MLP | 30,529 | 29.256 ± 0.015 | 0.042 ± 0.032 |
| 5000 | GIN | 30,532 | 6.569 ± 0.100 | 0.942 ± 0.002 |
| 5000 | MLP | 30,529 | 29.263 ± 0.015 | 0.022 ± 0.048 |

| N | gap = RMSE_GIN − RMSE_MLP | sd GIN | sd MLP | |gap| / sd GIN | |gap| / sd MLP | |gap| / pooled sd | sd of paired diffs | paired t |
|---|---|---|---|---|---|---|---|---|
| 500 | -13.430 | 10.602 | 0.015 | 1.3 | 873.6 | 1.8 | 10.596 | -2.8 |
| 5000 | -22.693 | 0.100 | 0.015 | 227.5 | 1474.5 | 318.0 | 0.109 | -464.2 |

Reference points on the fixed test set: sd(y_test) = 29.253 (RMSE of predicting the mean); best possible connectivity-blind predictor given the true per-node f (E_topology[c]·f) has RMSE = 28.531; topology alone, E[y | topology], has RMSE = 7.492; with topology and f both known, y is deterministic (floor 0).
