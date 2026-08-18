# agent-osmosis

Cross-model learning transfer for expedited fine-tuning. Instead of copying weights
between models (which doesn't work — different architectures, different weight
spaces), agent-osmosis extracts a portable **recipe** of *how* a task was learned and
replays it onto a different model, which then reaches target accuracy with far less
data than training from scratch.

A recipe is four things: the small subset of examples that carried the signal, the
curriculum order to teach them in, a failure→fix log, and the hyperparameters — plus
dated eval receipts that make it a trust artifact, not just a blob.

This repo is a **working MVP**: a real engine, three reproducible experiments, and a
dashboard. It runs on CPU in seconds with no model downloads. The models are small
classifiers standing in for LLMs; the *transfer machinery* is the point, and it's
written so a real `transformers` + `peft` model drops in behind the same interface.

## What it demonstrates (all reproducible)

| Experiment | Claim | Result on the demo substrate |
| --- | --- | --- |
| **Warm vs cold** | A recipe gets a *different* model to target with less data | **1.7× fewer** labeled examples (110 vs 188) |
| **Invariance** | Different base models agree on which examples matter | overlap **0.64 vs 0.09** random → **7× above chance** |
| **Compounding** | The recipe improves as more models use it | quality 0.68→0.80, cost 170→87 across 4 generations |

The invariance result is the load-bearing one: it's the assumption the whole idea
rests on, and on this substrate it holds.

**Validated on real transformers too.** `osmosis/torch_backend.py` reruns the mechanics on
actual multi-head-attention transformers (built from scratch in PyTorch — no downloads). There,
warm-start transfers even more strongly (**2.9× less data**), but strict influence-invariance
across *different architectures* turns out **weak** (~1.5× chance, stable across training length).
The honest, sharper conclusion: what transfers is the recipe's **difficulty-coverage structure**,
not the exact influence ranking — which is why a coverage recipe still warm-starts a different
architecture well. See `docs/technical.md` §8.

## Quickstart

```bash
pip install -e ".[viz]"          # engine + dashboard deps, CPU only

python examples/01_warm_vs_cold.py       # the speedup, + saves a plot
python examples/02_compounding.py        # the recipe compounding
python examples/03_influence_invariance.py   # the load-bearing experiment

streamlit run dashboard/app.py   # the control panel
pytest -q                        # the claims, as tests

pip install -e ".[llm]"                     # torch (CPU, no model downloads)
python examples/04_real_transformer.py      # the same ideas on real transformers
```

## The dashboard

A human-in-the-loop control panel (`dashboard/app.py`):

- **Warm vs Cold** — pick source/target models, run the transfer, watch the curves and speedup.
- **Recipe inspector** — read the actual coreset examples in curriculum order; **approve or reject** each failure→fix entry before it compounds; export the recipe as JSON.
- **Compounding** — run successive generations and watch quality climb / cost fall.
- **Invariance** — the cross-model agreement heatmap.

## How it works

```
osmosis/
  data.py        templated, human-readable text tasks (no downloads)
  models.py      OsmosisModel = frozen encoder (the "base model") + trainable head
  influence.py   which examples matter — class-balanced, margin-stratified coverage
  curriculum.py  easy → hard ordering
  recipe.py      the Recipe object: coreset + order + failure→fix + eval receipts + freshness
  transfer.py    extract a recipe from model A; warm-start model B with it
  experiment.py  warm-vs-cold, compounding, invariance
```

Two things we learned building it, both baked in:
- A good teaching coreset needs **coverage across difficulty**, not the easiest or
  hardest examples alone — either extreme plateaus well below the ceiling.
- Naive warm-starting can *hurt*; the **shrink-perturb** fix is the recipe's first
  failure→fix entry (toggle it in the dashboard).

## The trust layer (and the promptaura tie-in)

A recipe carries dated eval receipts and a `freshness()` function ported directly
from [promptaura](https://github.com/rishvaiyer)'s `freshnessOf` — the same
verification / reputation / freshness mechanic, one layer down at the weights. The
natural next step is a **verified recipe registry**: publish a recipe, and it carries
"verified to transfer to model X at score Y on date Z," with a transfer-confidence
score and a freshness badge. That trust layer — not the transfer math, which labs are
commoditizing — is the defensible part. See `docs/strategy.md`.

## The real-LLM path

The demo substrate is deliberately tiny so everything is reproducible in seconds. To
run it on a real transformer: implement the `OsmosisModel` interface over
`transformers` + `peft` (DistilBERT-LoRA fits CPU; Qwen2.5-0.5B for the generative
flex), swap the influence proxy for TracIn / DataInf, and the experiments run
unchanged. The influence-invariance study on two *different* transformers is the first
result worth publishing.

## Docs

- [`docs/eli5.md`](docs/eli5.md) — the idea explained simply, with diagrams
- [`docs/strategy.md`](docs/strategy.md) — landscape, differentiation, the business case
- [`docs/feedback.md`](docs/feedback.md) — an honest assessment of whether it's worth building
