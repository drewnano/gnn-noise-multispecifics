# Leave-one-topology-out (trispecific)

Pool 5000 molecules minus the held-out topology, 5 seeds, 9000-molecule test set. Clean labels. sd(y) over the whole test set = 20.08.

The MLP + topology embedding row for a held-out format is never trained, so its failure there is by construction. The question is whether the GIN beats `train mean` (no topology knowledge) and how close it gets to `oracle` (that topology's own true mean).

## Held-out topology: test RMSE

| held out | n test | sd(y) in fold | train mean | oracle | GIN | MLP (blind) | MLP + topology |
|---|---|---|---|---|---|---|---|
| 0 | 961 | 6.96 | 13.04 | 6.95 | 15.16 ± 10.96 | 13.26 ± 0.60 | 23.27 ± 17.81 |
| 1 | 992 | 7.64 | 11.91 | 7.64 | 8.44 ± 0.91 | 11.78 ± 0.29 | 12.80 ± 3.46 |
| 2 | 982 | 9.31 | 45.78 | 9.31 | 51.25 ± 6.76 | 45.76 ± 0.22 | 58.41 ± 6.82 |
| 3 | 1040 | 9.07 | 26.74 | 9.06 | 43.64 ± 3.73 | 26.70 ± 0.61 | 26.66 ± 10.54 |
| 4 | 1048 | 7.45 | 9.84 | 7.44 | 18.35 ± 2.27 | 9.69 ± 0.37 | 17.97 ± 4.96 |
| 5 | 1010 | 4.91 | 24.04 | 4.91 | 42.11 ± 14.98 | 24.27 ± 0.50 | 45.11 ± 24.34 |
| 6 | 999 | 5.92 | 22.00 | 5.92 | 45.35 ± 12.59 | 22.04 ± 0.38 | 37.37 ± 23.85 |
| 7 | 981 | 6.02 | 11.10 | 6.02 | 9.66 ± 2.40 | 11.15 ± 0.84 | 15.24 ± 13.40 |
| 8 | 987 | 6.28 | 6.63 | 6.28 | 36.75 ± 9.44 | 6.73 ± 0.31 | 14.01 ± 11.30 |

Mean over folds — GIN 30.08, blind MLP 19.04, MLP+topology 27.87; train mean 19.01, oracle 7.06.

## Held-out topology: Spearman

`nan` means the model predicted a constant on that fold, so rank correlation is undefined. That is the blind MLP's normal state here: it sits at predict-the-mean.

| held out | GIN | MLP (blind) | MLP + topology |
|---|---|---|---|
| 0 | 0.36 ± 0.14 | nan ± nan | 0.22 ± 0.33 |
| 1 | 0.28 ± 0.12 | nan ± nan | 0.14 ± 0.23 |
| 2 | 0.35 ± 0.03 | nan ± nan | 0.36 ± 0.04 |
| 3 | 0.22 ± 0.05 | nan ± nan | 0.13 ± 0.13 |
| 4 | 0.38 ± 0.05 | nan ± nan | 0.21 ± 0.08 |
| 5 | 0.37 ± 0.04 | nan ± nan | 0.29 ± 0.08 |
| 6 | 0.10 ± 0.08 | nan ± nan | 0.06 ± 0.20 |
| 7 | nan ± nan | nan ± nan | 0.32 ± 0.07 |
| 8 | 0.32 ± 0.13 | nan ± nan | 0.26 ± 0.08 |

## Seen topologies (same trained model): test RMSE

| held out | GIN | MLP (blind) | MLP + topology |
|---|---|---|---|
| 0 | 6.78 ± 0.11 | 20.61 ± 0.01 | 7.02 ± 0.16 |
| 1 | 6.84 ± 0.23 | 20.73 ± 0.01 | 6.89 ± 0.13 |
| 2 | 6.43 ± 0.11 | 14.68 ± 0.00 | 6.52 ± 0.15 |
| 3 | 6.29 ± 0.05 | 19.29 ± 0.01 | 6.80 ± 0.15 |
| 4 | 6.72 ± 0.15 | 20.91 ± 0.01 | 7.10 ± 0.08 |
| 5 | 6.88 ± 0.05 | 19.64 ± 0.01 | 7.17 ± 0.18 |
| 6 | 6.75 ± 0.10 | 19.81 ± 0.01 | 7.26 ± 0.11 |
| 7 | 9.73 ± 6.21 | 20.85 ± 0.01 | 7.12 ± 0.21 |
| 8 | 6.85 ± 0.09 | 21.06 ± 0.01 | 7.15 ± 0.14 |
