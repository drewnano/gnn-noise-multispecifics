# Does the GNN in this benchmark beat a parameter-matched MLP?

Three experiments on the synthetic benchmark from [arXiv:2601.23212](https://arxiv.org/abs/2601.23212),
*Disentangling multispecific antibody function with graph neural networks*. All settings follow the paper's
Table 1 wherever the paper and this repo disagree; the disagreements are catalogued in
[PAPER_VS_REPO.md](PAPER_VS_REPO.md).

**Summary.** Figure 2 reproduces: the GIN beats a connectivity-blind MLP by a wide margin, and the margin
grows with dataset size. But that margin is entirely topology identification, and nine learned scalars
capture it — a format-conditioned MLP ties the GIN on clean data, matches or beats it under every noise
model tested, and the GIN does worse than predicting a constant on topologies it has not seen. In this
benchmark, as configured, message passing has no measurable advantage over being told the format.

---

## 1. What the data actually is

Worth stating up front, because it determines what any model can learn.

Each molecule is a graph: 3 Fab binding domains (trispecific) plus 2 Fc nodes, with physical linkages as
edges. Each Fab carries an amino-acid sequence; an Ehrlich test function maps that sequence to a scalar
`f`. The label is a readout over the graph (`complex_format_fn` in `utils.py`):

```
y = (1/n) · [ w⊙f + A(w⊙f) + A²(w⊙f) ]
```

Three facts follow, all verified in `baseline/phi_check.py` and `baseline/train_baseline.py`:

1. **Φ is linear in `f` for a fixed topology.** Collapsing the sum gives `y = c_A · f`, where `c_A` is a
   per-format coefficient vector (`node_coefficients`). There are 9 trispecific topologies, so there are
   9 such vectors.
2. **Topology alone explains ~88% of the variance in `y`** at motif length 3. Predicting `E[y | topology]`
   gives RMSE 7.00 against sd(y) = 19.84.
3. **Fab *i* always occupies node *i*.** In `synthetic_dataset/_synthetic_dataset.py:105`, `stacked_seqs[idx].T`
   fixes the domain-to-node assignment; only the topology is redrawn per sample. Each Fab's sequences are
   mutations around its own Ehrlich function's initial sequence at `edit_prob=0.3`, so the three domain
   populations differ systematically in composition and **a node's identity is readable from its own
   features, with no message passing.**

Point 3 matters for interpreting everything below. A model that knows the format, knows which domain it is
looking at, and can estimate `f` has everything needed to compute `y`. Message passing is not required for
any of those three, which bounds what the GNN can contribute in-distribution before a single model is trained.

## 2. Setup common to all three experiments

| | |
|---|---|
| Format | trispecific (3 Fabs, 9 topologies) |
| Node encoder | mean of residue embeddings (paper Eq. 6) |
| Models | GIN; connectivity-blind MLP; format-conditioned MLP ("MLP + topology") |
| Params | 55,300 / 55,297 — matched by construction |
| Hyperparameters | dim 64, 3 layers, multiplier 2, Adam, lr 1e-3, batch 16, 50 epochs (paper Table 1) |
| Data | motif length 3, 10 motifs, edit prob 0.3 (paper Table 1) |
| Split | fixed held-out test set; val = 10% of the training pool |
| Seeds | 5 per cell; gaps are paired across seeds |

**Three models, not two.** The paper's Figure 2 compares the GIN to a *blind* MLP (§3.1: "the
connectivity-unaware MLP baseline"; §5.3: "blind to the graph connectivity A"). §5.3 describes a third,
*format-conditioned* MLP that receives "connectivity information as a learnable embedding from the one-hot
encoded format, which is concatenated to each node embedding," and §3.1 reports it performs "comparably to
the GNN." Both are included here, because the blind MLP sits at predict-the-mean in nearly every condition
and a flat baseline cannot measure degradation.

The conditioned MLP's topology input is `nn.Embedding(9, 1)` — **nine learned scalars in total**, one per
format, broadcast to every node and concatenated (node features go 64 → 65). This mirrors the repo's
`ConnectivityEmbedding(format, 1)`.

**Two deviations from the repo, both deliberate**, and both to follow the paper rather than the code:

- The repo's `mlp.yaml` and `gin.yaml` both set `use_connectivity_embedding: true`, so the repo as shipped
  does not reproduce Figure 2 — its MLP is the conditioned variant. Figure 2 here uses the blind MLP.
- The repo's `GINConv` MLP contains a LayerNorm not described in the paper. With the paper's mean encoder it
  makes the GIN collapse to predict-the-mean in ~4/5 seeds at every N. Ablated at N=2000: 1/3 seeds learned
  with it, 3/3 without. It is off by default here; the collapsed runs are kept in
  `results/fig2/collapsed_layernorm/`.

---

## 3. Experiment 1 — Figure 2 reproduction

**Method.** GIN vs blind MLP at N = 100 … 20,000, 5 seeds, plus additive Gaussian label noise at
0, 0.1, 0.25, 0.5, 1 and 2 × sd(y) applied to training labels only. 240 jobs.
Code: `train_baseline.py`; figure: `plot_fig2.py`; results: `results/fig2/`.

**Result.** The qualitative claim reproduces. The blind MLP is flat at 19.84 — predict-the-mean — at every
N, exactly as point 3 of §1 predicts it must be, since it cannot see topology and topology is 88% of the
variance. The GIN starts learning at N = 500 and reaches 6.19 at N = 20,000, below the topology-only
reference of 7.00. The gap grows with data.

**Noise.** The gap washes out at 0.5 × sd(y) for N = 500, at 1.0 for N = 1,000, and at ~2.0 for N ≥ 2,000.
The mechanism is not that noise raises the attainable error — at N = 20,000 a learning run goes from 6.19 to
only 6.78 at noise 1.0 — but that it prevents the GIN from escaping the predict-the-mean plateau within 50
epochs. Unbiased, independent noise is equivalent to a reduction in effective sample size (`N · signal/(signal+noise)`),
which is why more data pushes straight through it.

That last point motivated experiment 2: noise that *cannot* be trained away.

## 4. Experiment 2 — assay noise that does not average out

**Method.** Three observation models, applied identically to training, validation and *observed test* labels,
since a real dataset's test labels are measured by the same assay:

| family | model | levels |
|---|---|---|
| fold error | `y · exp(σ·N(0,1))` | σ = 0.2 (1.5-fold), 0.35 (2-fold) at 95% |
| batch effects | 9 campaigns with gain `exp(0.2·N(0,1))` and offset `0.5·sd(y)·N(0,1)`; a molecule is measured in its own topology's campaign with probability ρ | ρ = 0, 0.5, 1 |
| saturation | `median + h·tanh((y−median)/h)` | h = 1, 0.5, 0.25 × sd(y) |

ρ = 1 means campaign is perfectly confounded with format — the realistic failure mode when each format is
produced and assayed as its own campaign. Every noisy condition also carries 1.5-fold error on top.

Each run is scored twice: against the **true** test labels, and against **observed** test labels from the same
campaigns, which is what a random split of a real dataset would report. 9 conditions × 6 sizes × 5 seeds ×
3 models = 810 model runs. Code: `noise_study.py`; figure: `plot_noise_study.py`; results: `results/noise_study/`.

**Results at N = 5,000, RMSE against true labels** (sd(y) = 19.84, topology-only = 7.00):

| condition | GIN | MLP (blind) | MLP + topology |
|---|---|---|---|
| clean | 6.48 | 19.85 | 6.63 |
| 1.5-fold error | 7.59 | 19.95 | 7.60 |
| 2-fold error | 11.55 | 20.69 | 10.06 |
| batch ρ=0 | 13.19 | 20.14 | **8.30** |
| batch ρ=0.5 | 13.47 | 20.18 | **11.95** |
| batch ρ=1 | 18.85 | 20.26 | 19.38 |
| saturation 1 sd | 16.42 | 19.91 | **11.59** |
| saturation 0.5 sd | 20.04 | 20.04 | **15.23** |
| saturation 0.25 sd | 20.11 | 20.10 | **17.78** |

Three findings.

**(a) Clean data: the GIN and the conditioned MLP tie** (6.48 vs 6.63). This is a direct confirmation of the
paper's own §3.1 remark. The GIN's entire advantage over the blind MLP is topology identification, and nine
scalars reproduce it.

**(b) Format-confounded batch effects create a bias floor, and a random split hides it.** At ρ = 1, scored the
way a random split would score it — on observed test labels — the conditioned MLP posts 20.86 against the
blind MLP's 36.01, which reads as a ~25% improvement. Scored against truth, all three models are at
predict-the-mean. The model has learned the batch, the test labels carry the same batch, and the evaluation
rewards it. More data does not help; this is bias, not variance.

**(c) Saturation preserves ranking but destroys calibration.** At h = 0.5 sd the conditioned MLP keeps
Spearman 0.86 against truth while its RMSE is 15.23. If the downstream decision is "pick the top 20 to make,"
this is nearly harmless; if it is "predict the titer," it is fatal.

**On the GIN's fragility under noise — a narrower claim than it first appears.** Validation MSE on standardized
observed labels (1.0 = predicting the mean), measured on the model's *own* training distribution, separates
optimization failure from generalization failure:

| condition (N=5,000) | GIN | MLP + topology |
|---|---|---|
| clean | 0.107 | 0.111 |
| 1.5-fold error | 0.555 | 0.557 |
| batch ρ=1 | 0.344 | 0.344 |
| saturation 0.5 sd | **1.011** | 0.878 |
| saturation 0.25 sd | 1.003 | 0.956 |

Under saturation the GIN never fits anything — 1.011 is *worse* than predicting the mean, with its best
checkpoint at epoch 19 of 50 — while the conditioned MLP reaches 0.878 on identical labels. 0 of 5 GIN seeds
escaped the plateau at h = 0.5 or 0.25, at any N. But under batch ρ = 1 the two are *identical* at 0.344: the
GIN optimizes perfectly well there, onto the artifact, exactly as the MLP does. And under fold error the GIN
trails at N = 1,000 (0.734 vs 0.486) and ties at N = 5,000.

So "the GNN is less robust to noise" is only supported for **saturating readouts**. The proposed mechanism —
that the GIN must amplify a subtle structural signal through three message-passing layers, and tanh compresses
the gradient hardest in the tails where that signal lives — is untested.

## 5. Experiment 3 — leave-one-topology-out

Experiments 1 and 2 leave one claim standing, and it is the paper's own (§3.1): the conditioned MLP "is
restricted to the fixed set of architectures defined in the training set, whereas the GNN can learn
generalized topological rules that could theoretically extend to novel, unseen formats." This is the only
claim a 9-row lookup table cannot satisfy by construction.

**Method.** Each of the 9 topologies is held out in turn. Training molecules come only from the other 8
(pool 5,000 before the drop, ~4,440 after); each model is then scored on the unseen topology and, for context,
on the seen ones. Clean labels. 9 folds × 5 seeds × 3 models = 135 runs.
Code: `heldout_topology.py`; figure: `plot_heldout.py`; results: `results/heldout/`.

The conditioned MLP's embedding row for a held-out format never receives a gradient, so its failure there is
guaranteed and uninteresting. The question is whether the **GIN** beats two topology-free baselines, both
properties of the data rather than of any model:

- **train mean** — predicting the mean of the training labels (what you get knowing nothing about topology)
- **oracle** — predicting the held-out topology's own true mean (the floor for any format-level model)

**Result.**

| | mean over 9 folds |
|---|---|
| oracle | 7.06 |
| **train mean** | **19.01** |
| GIN | **30.08** |
| MLP (blind) | 19.04 |
| MLP + topology | 27.87 |

**The GIN beats the train-mean baseline in 2 of 9 folds.** On an unseen topology it is worse than ignoring
topology altogether, and worse than the conditioned MLP — whose embedding for that format is random and
untrained.

This is not undertraining: the same models score 6.3–6.9 on the seen topologies in eight of nine folds (fold 7
is 9.73 ± 6.21, where one seed collapsed; the mean over all folds is 7.03). The blind MLP's 19.04 tracks
the 19.01 baseline exactly, as it must, since it predicts a constant (in 23 of 45 runs literally so) — a useful
check that the fold bookkeeping is correct.

**Nor is the failure confined to structurally unusual formats.** Defining each fold's *novelty* as the distance
from the held-out topology's coefficient vector `c_A` to the nearest one in training:

```
Spearman(novelty, GIN excess over baseline) = 0.00
```

The worst fold is topology **8**, the *least* novel one — it has a near-twin in training (distance 0.66) and a
mean close to the global mean, so the baseline needs only 6.63. The GIN scores **36.75**. If the model had
learned anything resembling a topological rule, that is the fold where it should come nearly free.

Held-out Spearman is 0.30 for the GIN and 0.22 for the conditioned MLP, so some signal does transfer. But that
is the expected amount: within one topology `y = c_A · f` with `c_A` fixed, so any model that estimates `f` at
all ranks above chance. It is not evidence of topological generalization. What breaks is scale and offset.

---

## 6. Conclusions

1. **Figure 2 reproduces, and its gap is real** — but it measures topology identification against a baseline
   that cannot see topology, in a benchmark where topology is 88% of the variance.
2. **Nine learned scalars close that gap.** The format-conditioned MLP ties the GIN on clean data. The paper
   says this in §3.1; it is not a new finding, but it is not visible in Figure 2 either, and the repo's shipped
   config quietly makes *both* models conditioned.
3. **Under realistic assay noise the conditioned MLP matches or beats the GIN in every condition tested.**
4. **Format-confounded batch effects create an error floor that more data cannot fix, and the standard
   random-split evaluation actively conceals it.** This is the result with the clearest practical implication,
   and it is a property of the evaluation protocol, not of either architecture.
5. **The GIN does not extrapolate to unseen topologies** — worse than predicting a constant, with no
   relationship between failure and how novel the held-out format is.

Taken together: in this benchmark as configured, there is no measured regime in which message passing beats
being told the format.

## 7. Limitations

These are real and would change the picture:

- **The encoder is the binding constraint.** Mean-pooled residue embeddings (Eq. 6) recover `f` with R² ≈ 0.2–0.6
  from composition alone, versus ≈ 0.85–0.9 for a position-aware encoder (measured at motif length 4). With `f`
  largely unrecoverable, the sequence × position interaction — the one thing message passing uniquely offers —
  is invisible to *every* architecture. A position-aware encoder is the single highest-value change, though note
  §1 point 3 implies it should help the conditioned MLP at least as much as the GIN.
- **"Unseen format" here means an unseen connectivity within trispecific.** Generalizing across arity (train
  trispecific, test tetraspecific) is a different and arguably stronger test, and is not done here.
- **50 fixed epochs, 5 seeds.** This is exactly the regime where plateau escape dominates variance, so the GIN
  numbers are noisier than they look. The saturation result in particular is an optimization finding; a longer
  budget or a warm start could change it.
- **Figure 2's axis metric and gap sizes in the paper were not readable** from the HTML, so the reproduction is
  qualitative. Absolute numbers are not expected to match — the sequence set differs (§10 of PAPER_VS_REPO.md).
- **Single task instance.** One draw of Ehrlich functions (`TASK_SEED`), not averaged over tasks.

## 8. Reproducing

```bash
python -m baseline.train_baseline --help          # Figure 2 sweep
python -m baseline.noise_study                    # 810 runs, ~overnight on 8 cores
python -m baseline.heldout_topology               # 135 runs, ~20 min on 8 cores
python -m baseline.<script> --summarize           # rebuild tables from runs.jsonl
python -m baseline.plot_fig2 | plot_noise_study | plot_heldout
```

Every run is one line in the relevant `results/*/runs.jsonl`, carrying its own condition, size, seed and model,
so all tables and figures are regenerable. The study scripts resume: re-running skips completed cells.
