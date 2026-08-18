# Ship the recipe, not the weights

*A working experiment in cross-model learning transfer — and an honest account of where it
holds and where it doesn't.*

## The itch

Every time a better base model ships — which is now every few months — anyone who fine-tuned
the old one for a specific job has to redo that work from scratch. The knowledge the old model
gained doesn't carry over. That re-training tax is real and recurring. The obvious fix, "just
copy the fine-tuned weights into the new model," doesn't work: different architectures live in
different weight spaces, so a weight that means one thing in model A is noise in model B.

So the transfer has to happen a level up from weights. That's the premise of `agent-osmosis`:
extract a portable **recipe** of *how* a task was learned, and replay it onto a different model.

## What a recipe is

Not the trained model, and not its weights — the *training program*:

- the small **coreset** of examples that actually carried the learning signal,
- the **curriculum** order to teach them in (easy → hard),
- a **failure → fix log** (what the model kept getting wrong, and what fixed it),
- the **hyperparameters** that worked,
- and dated **eval receipts** so a recipe is a trust artifact, not an opaque blob.

A recipe is model-agnostic data. You can hand it to any architecture.

## Does it work?

On a fast CPU substrate (small models standing in for LLMs, so everything runs in seconds and
reproduces), a model that warm-starts from a recipe reaches target accuracy with **~1.7× less
labeled data** than training from scratch — and **~2.9×** when we rerun it on real from-scratch
PyTorch transformers with genuine attention and backprop. That warm-start win also beats plain
distillation and beats selecting only the "hardest" examples (which, it turns out, is reliably
*worse* than a balanced sample).

## The part most write-ups would hide

Two honest results, because they're the interesting ones:

**1. Our clever selection isn't (yet) the reason it works.** We select a coverage coreset —
class-balanced, spread across the difficulty spectrum. It gives a better *coreset-only* ceiling
than naive picks. But head-to-head on warm-start speed, a *random* class-balanced subset is a
noisy tie with it on this easy substrate. The durable win is "warm-start from a small balanced
set at all," not our particular curation. Establishing that coverage beats random needs a harder,
real task — so we're not claiming it until it's earned.

**2. Influence isn't invariant across architectures.** The whole "compute which data matters
once, reuse it everywhere" story rests on different models agreeing on which examples are
influential. On artificially-similar encoders they agree strongly (7× chance). On genuinely
different transformer architectures, agreement collapses to ~1.5× chance — and it's stable across
training length, so it's real, not noise. The reconciliation: what transfers across architectures
is the *difficulty-coverage structure* of a task, not the exact influential set. That's a sharper,
more defensible claim than the one we started with — and it came out of running the experiment.

## The one angle worth chasing

The transfer mechanism itself is crowded — Sakana's Text-to-LoRA, Cross-LoRA, LoRA-X, and Ramp's
PorTAL all move adaptations between models. The genuinely under-explored idea is **compounding**:
a recipe that *improves every generation* because each model that runs it folds back the failures
it still had. A frozen adapter depreciates; a recipe with a data-network-effect gets cheaper to
use over time. On the substrate, quality climbs and cost falls across four generations. Whether
that survives on a real Llama-2 → 3 → 4-style chain is the open question — and the only one here
that a well-funded lab hasn't already answered.

## Where this goes

- **The experiment that decides it:** a rigorous 3-generation study on real pretrained models. If
  a compounding recipe delivers >2× cumulative speedup across three model releases, that's a
  result worth publishing. If it's ≤1.3×, this is a clean portfolio piece and an honest negative
  result — which is still worth more than a confident wrong one.
- **What it isn't:** a product to sell. Fine-tuning is cheap and the platforms that own
  distribution are absorbing this fast. This is a research demonstration and an open engine, not
  a SaaS.

The code, the dashboard, and every number above are reproducible: `make demo`.
