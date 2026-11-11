# Paper vs. repo: Figure 2 (GIN vs MLP scaling)

Sources: paper text, arXiv:2601.23212v1 (methods, Table 1, Figure 2 caption, read from the HTML version);
repo at d6c2858 (`models/_conv.py`, `configurations/`, `wandb_sweeps/model_comparison_sweep.yaml`, `train.py`).
The Figure 2 images were not readable, so the axis metric and gap sizes are not known.

## Differences

| # | Item | Paper | Repo (Fig. 2 sweep, `run_sweep.py model`) | Impact |
|---|---|---|---|---|
| 1 | Node encoder | Mean of residue embeddings (Eq. 6) | 3 transformer blocks (1 head, rotary positions, `models/_conv.py`), then mean | **High.** The mean of embeddings sees composition only. f is ~0.2–0.6 R² predictable from composition vs ~0.85–0.9 position-aware (motif length 4). |
| 2 | MLP topology input | Fig. 2's MLP is blind: §3.1 "We compared the GIN against the connectivity-unaware MLP baseline", §5.3 "blind to the graph connectivity A". A **format-conditioned** MLP is a *separate, third* baseline (§5.3) that receives "connectivity information as a learnable embedding from the one-hot encoded format, which is concatenated to each node embedding"; §3.1 says it performs "comparably to the GNN". The paper never says whether it appears in Fig. 2 — the caption names only GIN vs MLP. | `use_connectivity_embedding: true` in **both** `gin.yaml` and `mlp.yaml` (the `MLP` class default is `False`) | **High.** The repo's shipped MLP config is the paper's format-conditioned variant, not Fig. 2's blind one, so the repo as-configured does not reproduce Fig. 2. With the embedding the MLP learns E[y \| topology], which is most of the variance; with our mean encoder the GIN gap is then ~0. |
| 3 | GIN topology input | Not mentioned | Learned per-topology scalar, concatenated to nodes | Medium |
| 4 | MLP depth | "update at each layer k" (same depth implied) | `num_layers - 1` node blocks (2 vs GIN's 3); block width ignores `multiplier` | Capacity mismatch: 118.7k vs 110.3k params at dim 64 |
| 4b | LayerNorm in node MLPs | Not mentioned (Eq. 7: "2-layer MLP") | `GINConv` MLP has LayerNorm (`models/layers/gnn.py`) | **High with the mean encoder.** GIN gets stuck predicting the mean in ~4/5 seeds at every N (100–20,000); without LN, 3/3 seeds learn at N=2000. The transformer's more varied node features probably avoid this in the repo. |
| 5 | Embedding dim | 64 | 64 in sweep (`gin.yaml` default is 32) | — |
| 6 | Epochs / stopping | 50 epochs; no early stopping or scheduler mentioned | `max_epochs: 30`, early stopping patience 10 (min_delta 1e-4), ReduceLROnPlateau (×0.1, patience 10), grad clip 1.0, best-val checkpoint for test | Medium: short training hurts GIN at small N (we saw plateau-escape in 30–100 epochs) |
| 7 | Batch size | 16 | 16 in sweep (`default.yaml` 32) | — |
| 8 | Motif length | 3 | 3 in sweep (`default.yaml` 4) | Changes data; sd(y) 29.3 → 19.8 on our test set |
| 9 | Split | 80/10/10 random | `fixed_test_size: 1000`; val = 10% of N | Minor |
| 10 | Sequences | Full paired OAS | Subset (README; `sequence_examples.csv`, PSSM pickles) | Unknown; numbers won't match exactly |
| 11 | Readout Φ | y = (1/n)\|w⊙f + A(w⊙f) + A²(w⊙f)\| | Sum of the same terms, no abs (`utils.py`) | None in practice: all terms are non-negative |
| 12 | Adjacency | Not specified | Each edge stored once as (min, max): A is directed/upper-triangular, and GIN gets one-way edges | Node numbering affects y and message passing |
| 13 | Label scaling | Not stated | Raw y into MSELoss | Optimization only |
| 14 | Label noise | None | `noise_level` on per-Fab f (DataModuleConfig default 0.01; yaml and sweep 0.0) | — |
| 15 | Seeds | Not stated | 5 seeds; `train.py` calls `seed_everything(cfg.seed)` | Repo runs are seeded via Lightning; our `seed_all` makes our own loop seeded |

## What `baseline/train_baseline.py` uses for the Figure 2 reproduction

These choices follow the paper text wherever the paper and repo disagree:
- mean encoder, blind MLP (`--no-conn-emb`), dim 64, `--multiplier 2` without LayerNorm, 3 layers, `--motif-length 3`, `--batch-size 16`, `--max-epochs 50`
- Our additions:
  - The MLP has the same depth and blocks as the GIN (params 55,300 vs 55,297).
  - The best-val checkpoint is used for test; no LR scheduler or clipping.
  - y is standardized on train.
  - Fixed 2000-molecule test set; val = 10% of N.

## Worth revisiting when modernizing

- **Encoder.** Swap in a position-aware encoder, or a frozen protein language model (pLM) embedding per domain. As it stands, the sequence signal is mostly invisible, so the benchmark mostly tests topology identification.
- **Edges.** Make edges undirected (symmetric A and both directions to GIN), or document the directionality as intended.
- **Report three models.** Blind MLP, topology-conditioned MLP and GIN. Blind vs conditioned isolates topology identification; conditioned vs GIN isolates sequence × position interaction.
- **Variance decomposition.** Report how much of the variance in y topology alone explains. In the current settings it is ~88–93%.
