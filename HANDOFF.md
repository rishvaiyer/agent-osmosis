# HANDOFF — agent-osmosis

A complete state-of-project doc so another agent (Codex, etc.) can take over. Read this
top to bottom before touching anything.

## What this project is (one paragraph)

`agent-osmosis` demonstrates **cross-model learning transfer for expedited fine-tuning**:
you can't copy weights between models (different architectures, different weight spaces),
so instead we extract a portable **recipe** of *how* a task was learned — the small
coverage-coreset of examples that mattered, their easy→hard curriculum order, a
failure→fix log, and hyperparameters — and warm-start a *different* model with it. It
reaches target accuracy with meaningfully less labeled data. A recipe also **compounds**:
each model that runs it folds back the failures it found, so it improves generation over
generation. There is a trust layer (dated eval receipts + a `freshness()`/confidence score).

## Current state (as of this handoff)

Working MVP, pushed to `main` at https://github.com/rishvaiyer/agent-osmosis.

- **Engine** (`osmosis/`): data, models, influence, curriculum, recipe, transfer,
  experiment, metrics, baselines, torch_backend. Pure NumPy/scikit-learn default path
  runs the full demo in seconds on CPU, no downloads.
- **Real transformers** (`osmosis/torch_backend.py`): from-scratch PyTorch multi-head
  attention models (no downloads) — warm-start works (2.9× less data); influence-
  invariance across architectures is weak (~1.5× chance).
- **Dashboard** (`dashboard/app.py`): Streamlit, 4 tabs, animated warm-vs-cold race,
  human approve/reject on the failure→fix log, recipe inspector, invariance heatmap.
- **Examples** (`examples/01..06`): warm-vs-cold, compounding, invariance, real
  transformer, baselines, and a ready-to-run real-**pretrained**-model script (06, needs
  network + `.[llm]` extras — does NOT run in a no-download sandbox).
- **Tests** (`tests/test_engine.py`): 7 passing (`pytest -q`).
- **Docs** (`docs/`): eli5, technical (implementation-level), strategy, feedback.

## The honest scientific picture (do not overclaim)

1. **Robust wins:** warm-start from a small class-balanced coreset + curriculum beats
   cold start (~1.7× linear, ~2.9× real transformers), beats distillation, and beats
   EL2N-hardest-only (which is reliably *bad*).
2. **NOT established:** that our "coverage" selection beats a *random* class-balanced
   subset — on the toy substrate it's a noisy tie. This needs a harder / real task.
3. **Key finding:** influence-invariance is architecture-sensitive. What transfers across
   architectures is the *difficulty-coverage structure*, not the exact influential set.
4. The transfer math is crowded (Sakana Text-to-LoRA, Cross-LoRA, LoRA-X, PorTAL). The
   less-crowded angles are the **compounding recipe** and the **cross-architecture
   invariance study**. Frame around those, not "cross-model transfer" (name is taken).

## How to run

```bash
pip install -e ".[viz]"                 # engine + dashboard (CPU, no downloads)
python examples/01_warm_vs_cold.py      # + 02..05
streamlit run dashboard/app.py
pytest -q

pip install -e ".[llm]"                 # torch, for from-scratch real transformers
python examples/04_real_transformer.py

# the real, publishable experiment — RUN LOCALLY (needs network):
pip install transformers peft datasets accelerate
python examples/06_real_pretrained.py --n 400 --model bert-tiny
```

## Architecture / interfaces (where to plug in)

- `OsmosisModel` (models.py) is the swap point: `train_stream / warm_prefit / predict /
  margins / per_example_loss`. `torch_backend.TorchModel` implements the same surface for
  real transformers. Implement it over `transformers`+`peft` for real pretrained models.
- `Recipe` (recipe.py) is model-agnostic and serializes to JSON — the portable artifact.
- Influence is pluggable: `influence.py` (linear proxy), `torch_backend.el2n_scores`,
  and `examples/06` has a real **TracIn** implementation to promote into the package.

## Next steps, prioritized (highest leverage first)

1. **Run `examples/06` on real pretrained models with TracIn** (local, has network). Does
   TracIn-influence agree across architectures more than EL2N (~1.5×)? This is the
   headline result. Promote a `osmosis/influence_tracin.py` from the example once validated.
2. **Real dataset** (SST-2 / AG News) as a first-class task alongside the templated one,
   so claims aren't on toy data. `examples/06` already loads SST-2.
3. **Establish coverage-vs-random** on a genuinely hard task, or retire the "coverage is
   the secret sauce" claim and lead with "compounding recipe" instead. Be honest.
4. **Reproducibility polish:** `Makefile` (`make demo`), a Colab notebook, a results table
   with CIs in the README, a tests badge, a 20s dashboard GIF.
5. **Writeup** (`docs/writeup.md`): the honest arc — hypothesis → built it → surprising
   invariance result → the sharper claim that survived. Reframe around compounding recipe.
6. **Trust/registry layer** (optional product angle): a small API to publish/consume
   recipes with transfer-confidence + freshness. The one defensible, uncrowded piece.

## Guardrails / gotchas

- Keep the CPU-no-download default path fast and green — it's the demo.
- Everything is seeded; keep it reproducible. Report CIs, not single numbers.
- Naive warm-start can hurt generalization → shrink-perturb is the fix (transfer.py,
  dashboard toggle). It's also logged as the recipe's first failure→fix entry.
- Do NOT overclaim. The strength here is a working system + honest experiments, not a
  novel algorithm. Match the docs to what the numbers actually support.

## Commit conventions

Small, descriptive commits. Run `pytest -q` before pushing. Branch is `main`.
