# agent-osmosis — Technical Explainer

A precise walk through what the engine actually computes. Pairs with the plain-language
[`eli5.md`](eli5.md). Everything here is implemented in `osmosis/` and exercised by
`tests/` and `examples/`.

---

## 0. Problem statement

Given a task `T` learned by a source model `A`, produce a portable artifact `R` (a *recipe*)
such that a different model `B` trained with `R` reaches a target validation accuracy `α`
using fewer **unique labeled examples** than training `B` from scratch. Weights are not
transferable across `A` and `B` (distinct parameter spaces), so `R` is defined entirely in
*data/schedule space*, which is architecture-agnostic.

We measure cost as unique labeled examples consumed, not wall-clock or epochs — labeled data
is the scarce resource, and it makes warm vs cold an apples-to-apples comparison.

---

## 1. Substrate (`models.py`, `data.py`)

An `OsmosisModel` factorizes into a **frozen encoder** `φ_m` (the base model's fixed
representation) and a **trainable head** `w` (what fine-tuning moves):

```
score(x) = ⟨w, φ_m(x)⟩ ,   ŷ = 1[score(x) > 0]
```

- `φ_m` is a `HashingVectorizer` whose config (n-gram range, dimensionality, signed hashing)
  differs per registered base model `m ∈ {osmo-base-a, -b, -c, wide-d}`. Different encoders
  are the stand-in for different architectures: the same text lands at different coordinates,
  which is exactly why raw `w` doesn't transfer but a recipe can.
- The head is an `SGDClassifier(loss="log_loss")` trained by `partial_fit` over mini-batches,
  so we can read validation accuracy after every `eval_every` examples and build a genuine
  step→accuracy `Curve`.

Tasks are templated binary text classification. An example's latent difficulty is controlled:
*easy* examples use strong polarity words (`"the export works perfectly"`); *hard* examples are
hedged with mild, same-polarity words (`"honestly the export is fine"`) — learnable but
data-hungry, which is what gives the recipe something to be smart about. Ceiling ≈ 0.89.

The whole `OsmosisModel` interface is ~40 lines; §7 swaps it for a real transformer.

---

## 2. Which examples matter — influence proxy (`influence.py`)

We need two things from influence: a **teaching coreset** (to build the recipe) and a
**ranking** (to test cross-model agreement). Both come from a lightly-trained *probe*
`φ_m`-model `p` (3 shuffled passes over the pool).

Per example `i` with margin `μ_i = ⟨w_p, φ_m(x_i)⟩` and correctness `c_i = 1[ŷ_i = y_i]`:

**Informativeness score** (used for the ranking / invariance test):
```
s_i = norm( 1 / (1 + |μ_i|) ) · (c_i ? 1.0 : 0.3)
```
Near-boundary, correctly-labeled examples score highest; likely-mislabeled points are damped.

**Teaching coreset** (used to build the recipe) is *not* a top-k of `s_i` — that was a bug we
hit and fixed. Training only on the most-informative (near-boundary) points plateaus ~0.57;
training only on the most-confident prototypes plateaus ~0.61. What reaches the ceiling with
~15% of the data is a **class-balanced, margin-stratified coverage sample**: within each class,
sort by `|μ_i|` and take an even `linspace` across the whole spectrum. This spans easy
prototypes (establish the boundary) and hard cases (teach the subtle features). Empirically a
66-example (15%) coverage coreset reaches 0.87 vs the 0.89 full-data ceiling — the LESS-style
"few percent matches full accuracy" claim, reproduced on this substrate.

> This is a proxy for TracIn / influence functions / DataInf / LESS. `recipe.method` records
> which estimator produced a given recipe; §7 swaps in a real one.

---

## 3. The recipe object (`recipe.py`)

```python
Recipe = {
  influence_set : list[int]          # coreset (indices into the train pool)
  ordering      : list[int]          # curriculum order over the coreset (easy→hard)
  failure_fix_log : list[FailureFix] # {failure_text, true_label, fix_indices, note}
  hyperparams   : dict               # lr, batch, prefit_passes
  eval_receipts : list[EvalReceipt]  # {target_model, reached_acc, steps, dated, outcome}
  method, version, created_at, updated_at
}
```

Curriculum ordering (`curriculum.py`) sorts the coreset easy→hard by a probe's per-example
logistic loss.

**Trust layer.** `freshness()` is ported line-for-line from PromptAura's `freshnessOf`:
`broken` if failed ≥ transferred receipts; `unverified` if none; `edited` if the recipe was
modified after its last successful transfer; else `fresh / aging / stale` by a 30/90-day
window. `transfer_confidence() ∈ [0,100]` is a weighted blend:
```
0.40·success_rate + 0.25·cross_model_breadth + 0.25·recency + 0.10·compounding_depth
```
These are the fields a *verified recipe registry* would index on.

---

## 4. Warm start and honest accounting (`transfer.py`)

`extract_recipe(T, A)`: train `A` on `T`; select coverage coreset (§2); order it (curriculum);
build the failure→fix log from `A`'s residual validation errors; record hyperparameters.

`warm_start(T, R, B)`:
1. **Pre-fit** `B` on the coreset, `prefit_passes` times, in curriculum order. Passes over the
   same small set are cheap and do **not** inflate the cost — cost is *unique* examples, so
   `seen := |coreset|` after pre-fit regardless of passes.
2. **Stream** the remaining pool (coreset examples *excluded*, so every streamed example is new),
   logging the curve, until the pool is exhausted.
3. Emit an `EvalReceipt` with `steps_to_target(α)`.

`cold_start(T, B)` streams the full pool from scratch. Because warm's pre-fit examples are
counted and its stream excludes them, `steps_to_target` is a fair unique-example comparison.

**Shrink-perturb.** Naive warm-starting can hurt generalization (Ash & Adams, 2020). The fix —
`w ← ρ·w + σ·ε` (ρ=0.6, σ=0.02) — is the recipe's first failure→fix entry and a dashboard
toggle; a live demonstration of the failure→fix mechanic applied to the machinery itself.

---

## 5. The three experiments (`experiment.py`)

**Warm vs cold.** `T` fixed, extract `R` from `A`, run `warm_start`/`cold_start` for `B` over
`trials` seeds; average curves on a shared x-grid; report mean ± 95% CI steps-to-target and
`speedup = cold_mean / warm_mean`. Demo result: **110 vs 188 → 1.71×**.

**Influence invariance** (the load-bearing test). For each base `m`, train it on `T`, take the
top-k by `s_i` (§2). Report the pairwise Jaccard matrix, the mean off-diagonal overlap, and a
random-subset baseline `E[Jaccard]` of two random k-subsets of n. Verdict `invariant` iff mean
overlap `> 3×` baseline. Demo result: **0.642 vs 0.089 → 7.2× → invariant**. This is the
assumption the whole idea rests on; if it failed, recipes couldn't be model-agnostic.

**Compounding.** `T` fixed; start from a deliberately small recipe (5% budget). For each
generation (base) `g`: measure `recipe_quality` = accuracy a fresh model reaches from the
recipe *alone* (no streaming); record warm/cold cost; then `discover_fixes` (the base's residual
errors + same-class coverage examples not yet in `R`) and `compound` them back, bumping
`version`. Demo result over 4 generations: quality **0.68→0.80**, warm cost **170→87**. The
recipe improves with use — the property a one-shot adapter transfer lacks.

---

## 6. What the results do and don't show

- **Do:** the mechanism is coherent and the invariance premise holds on a controlled substrate
  where we can vary "architecture" (encoder) cleanly and run many seeded trials fast.
- **Don't:** these are hashing-encoder linear models, not transformers. The numbers are
  illustrative of the *machinery*, not benchmarks. The encoders are more similar to each other
  than two real model families are, so the 0.64 overlap is an optimistic proxy — which is
  exactly why §7 is the next milestone.

---

## 7. The real-LLM path

The interface is small on purpose. To run on real models, implement `OsmosisModel` over
`transformers` + `peft`:

- `φ_m` = a frozen pretrained encoder (DistilBERT 66M fits CPU; BERT-tiny 4M for <30s;
  Qwen2.5-0.5B for a generative flex). Head = a LoRA adapter (`get_peft_model`), which *is* the
  portable warm-start artifact.
- Influence: swap the proxy for **TracIn** (⟨∇ℓ(train_i), ∇ℓ(val)⟩ summed over checkpoints — no
  Hessian, cheap on a LoRA's few-k trainable params) or **DataInf** (closed-form, LoRA-native).
- **First result worth publishing:** the §5 invariance study across two *different* real
  transformers (e.g. DistilBERT vs BERT-tiny). Non-trivial overlap there is the empirical claim
  the strategy doc identifies as the defensible, uncrowded contribution.

Everything else — recipe object, warm/cold harness, compounding loop, trust layer, dashboard —
runs unchanged behind that interface.

---

## 8. Real-transformer validation (`torch_backend.py`)

We didn't stop at the linear substrate. `osmosis/torch_backend.py` reruns the mechanics on
**actual transformers** — multi-head self-attention, real backprop — built from scratch in
PyTorch (model hubs are unreachable from the sandbox, so nothing is downloaded). Four
`TORCH_BASES` are genuinely different architectures (depth/heads/width). Run it with
`python examples/04_real_transformer.py`. Two findings, and the second is the important one:

- **Warm-start transfers strongly.** A coverage-coreset + curriculum recipe warm-starts a
  *different* transformer to target with **~2.9× less data** (102 vs 296 examples). The
  transfer mechanism holds on real models.
- **Strict influence-invariance is architecture-sensitive — and weak.** Cross-architecture
  agreement on the top-EL2N examples is only **~1.5–1.8× chance** (mean Jaccard ≈ 0.13–0.17),
  *stable across 5/12/25 training epochs* so it's a real effect, not noise — versus ~7× on the
  linear substrate whose encoders were artificially similar.

**The reconciliation, and the actual thesis update:** what transfers across architectures is the
recipe's **difficulty-coverage structure**, not the exact influence ranking. The coverage coreset
spans easy→hard by each model's own margins, so it warm-starts a new architecture well *even when*
the two models' EL2N-top-k sets diverge. This directly refines the strategy doc's load-bearing
question: don't bet on "the same examples are influential for every model" (weak); bet on "the
*difficulty structure* of a task is shared, and a coverage recipe captures it" (holds here). That
is a sharper, more defensible claim — and it came out of actually running the experiment.

---

## References

Ash & Adams, *On Warm-Starting Neural Network Training* (NeurIPS 2020) ·
Pruthi et al., *TracIn* (2020) · Xia et al., *LESS* (2024) · Kwon et al., *DataInf* (ICLR 2024) ·
Hu et al., *LoRA* (2021). Landscape and citations in [`strategy.md`](strategy.md).
