| N (train+val) | noise (y) | model | params | test RMSE (mean ± sd) | Spearman (mean ± sd) |
|---|---|---|---|---|---|
| 20000 | 0.5 | GIN | 55,300 | 6.476 ± nan | 0.914 ± nan |
| 20000 | 0.5 | MLP | 55,297 | 19.835 ± nan | nan ± nan |
| 20000 | 1.0 | GIN | 55,300 | 7.118 ± nan | 0.889 ± nan |
| 20000 | 1.0 | MLP | 55,297 | 19.834 ± nan | nan ± nan |
| 20000 | 2.0 | GIN | 55,300 | 7.189 ± nan | 0.883 ± nan |
| 20000 | 2.0 | MLP | 55,297 | 19.836 ± nan | nan ± nan |

| N | noise | gap = RMSE_GIN − RMSE_MLP | sd GIN | sd MLP | |gap| / pooled sd | sd of paired diffs | paired t |
|---|---|---|---|---|---|---|---|
| 20000 | 0.5 | -13.359 | nan | nan | nan | nan | nan |
| 20000 | 1.0 | -12.717 | nan | nan | nan | nan | nan |
| 20000 | 2.0 | -12.648 | nan | nan | nan | nan | nan |

Reference points on the fixed (clean) test set: sd(y_test) = 19.839 (RMSE of predicting the mean); best possible connectivity-blind predictor given the true per-node f (E_topology[c]·f) has RMSE = 18.866; topology alone, E[y | topology], has RMSE = 6.998; with topology and f both known, y is deterministic (floor 0).
