# model-osmosis — Strategy

*Synthesis of three research sweeps (differentiation, monetization, new layers), Aug 2026.
Candid, sourced, not hype. This is the map, not the marketing.*

---

## The one-sentence thesis

> **You're not transferring weights. You're building the trust, freshness, and provenance
> layer for a world where weight-level transfer is already commodity — the PromptAura thesis,
> one layer down at the weights.**

Everything below flows from that.

---

## The uncomfortable landscape check

Most of the "obvious" ideas here are **already shipped**. Naming them so we don't reinvent:

| Idea | Status | Owner |
| --- | --- | --- |
| Hypernetwork emits adapter from task description | ✅ shipped | Sakana — Text-to-LoRA / Doc-to-LoRA |
| Data-free cross-architecture LoRA transfer (SVD subspace align) | ✅ shipped | Cross-LoRA, LoRA-X, Trans-LoRA |
| **Hypernetwork emits adapter for an *unseen base model*** | ✅ **shipped Jul 2026** | **Ramp Labs — PorTAL** (~98% of per-task-LoRA on unseen bases vs ~14% for Cross-LoRA) |
| HP-schedule transfer across scale | ✅ solved | μP / CompleteP |
| Task-vector transport across models | ✅ shipped | few-shot orthogonal alignment, gradient-sign masking |
| Federated LoRA machinery | ✅ mature | SDFLoRA, PreLort |

**Implication:** "Angle A" (unseen-base hypernetwork), which we thought was open, is **PorTAL's turf**. Pursuing it as framed makes us a strictly-dominated fast-follower.

**The landmine:** a 2026 paper — *"On the Importance of Trivial Baselines: Re-evaluating LoRA
Adapter Transfer"* — shows **naively copying an adapter beats the fancy transfer methods for
related models.** Any transfer claim must beat the copy baseline AND justify itself on the hard
regime (distant families, different tokenizers, generative/MT tasks) or it's dead on arrival.

---

## Where the actual whitespace is: Angle B — the *recipe*, not the weights

Everyone transfers **artifacts** (data, weights, adapters). **Nobody transfers the *process*.**

Define a **Recipe** as a typed, versioned, model-agnostic artifact:

```
Recipe = {
  influence_set:   the ~5% of examples that actually drove learning,
  ordering:        the curriculum / sequence that converged fastest,
  failure_fix_log: discovered failure modes → the examples that fixed them,
  hp_schedule:     the hyperparameter schedule (μP handles the scale mapping),
}
```

with a defined **replay operator** that refits a Recipe onto *any* base model. This is
**program transfer, not weight transfer** — and it's genuinely under-occupied.

### Two things nobody has done
1. **A portable recipe object** replayed onto a new, architecturally different base to hit
   target accuracy in a fraction of a full fine-tune.
2. **Compounding across model generations** — the cure for what the literature calls AI's
   **"scientific amnesia"**: today's pipelines re-train each new base but *never accumulate
   knowledge about how to train the next one.* A recipe where Gen N's failure-fix log seeds
   Gen N+1's curriculum gets **cheaper and better every generation.** This is the moat.

### The single strongest "impressive-if-true" claim to aim for
> *Fine-tuning knowledge can be stored as a portable, model-agnostic recipe that, replayed
> onto a brand-new and architecturally different base, reaches target-task accuracy in under
> ~10% of full-fine-tune compute — and gets cheaper every model generation because the recipe
> compounds.*

It sidesteps PorTAL (process, not adapter), sidesteps the copy-baseline landmine (targets
heterogeneous bases + generative tasks), and has a killer figure no existing paper can show:
**cost-to-accuracy falling monotonically across a base-model chain.**

### The de-risking experiment to run FIRST (before building anything big)
**Are high-influence examples base-invariant?** Compute the top-k influential examples (LESS /
TracIn style) on several heterogeneous bases and measure correlation. If the "which data
matters" signal transfers, the whole thesis stands and you get a clean publishable result on
its own. If it doesn't, you learn that cheaply before over-building. **This is the pivotal
pivot-or-persevere test.**

---

## The layer stack (what to build, in order)

Ranked by (novelty × feasibility × impact), reusing PromptAura's DNA where possible:

### Build now — fast, defensible, commodity-composition (the trust layer)
1. **Transfer-freshness tracking** — continuously re-run each adapter/recipe against new base
   models + rolling post-cutoff evals; tag each as `still transfers / decayed / broken` with a
   decay curve. *Directly analogous to PromptAura's per-model prompt re-verification.* Fastest
   credible win; mostly plumbing + a maintained eval corpus.
2. **Verified recipe/adapter registry with reputation** — not another storefront (HF, LoRA
   Hub, AI Mod Store already exist and monetize adapters at ~$0). The unclaimed differentiator
   is the **verification + reputation + freshness layer on top**: every artifact carries
   "verified to transfer to model X at eval Y on date Z," creators ranked by how often their
   artifacts survive re-verification. This is PromptAura's mechanic, at the weights.
3. **Provenance receipts** — cryptographically-signed lineage per adapter/recipe: which source
   model+data it derived from, which transfer op produced it, eval receipts at each hop. As
   weight-transfer becomes routine, **licensing/data-contamination becomes the blocking
   enterprise concern** and there's no incumbent. Genuinely novel.

### Bet the research narrative on — the only place with a compounding moat
4. **Trajectory / curriculum transport** (Angle B core) — ship the ordered learning program,
   replay competence-adjusted onto B.
5. **Compounding recipes** — the living artifact that improves every pass. The actual flywheel.

### High-risk moonshots — gate them, start within-family
6. **Adaptation interlingua** — one shared latent for adaptations across families (O(N) hub vs
   O(N²) pairwise). Real upside, easy to over-claim; the universality evidence is partial.
7. **Recipe-conditioned adapter emission** — emit the adapter *from the transferred recipe*
   rather than from task text. This is how Angle A survives: as a downstream demo *of* Angle B,
   cleanly distinct from PorTAL, instead of a competitor to it.

---

## The money: feature, product, or company?

**Honest verdict: strong feature, plausible venture wedge, weak standalone company as first framed.**

- The pain is real and now quantified — Gartner-style estimates put model-deprecation rework at
  **15–25% of 3-yr enterprise AI cost.**
- But value accrues to whoever owns the fine-tuning + serving surface, and **incumbents are
  actively buying this exact IP**: CoreWeave bought OpenPipe (Sept 2025); Rubrik bought Predibase
  (~$100M+, Jun 2025); Sakana ($2.65B, $135M Series B) is *publishing the core tech as papers.*
- Fine-tuning has commoditized to **~$0.50/M training tokens, ~$10–100 per 7B LoRA run.** When
  the base op is that cheap, a "faster/cheaper" layer has almost no price umbrella.

### Buyers, ranked by willingness to pay
1. **Model hosting / inference platforms** (Together, Fireworks, Baseten, Modal, CoreWeave, Fal) —
   "instantly migrate every customer's adapter on new-base release day" is a retention feature.
   They buy or build. *Best OEM channel, but only ~5 buyers who could replace you.*
2. **Enterprises with 10s–100s of production adapters** (fintech, healthcare, legal, support,
   coding) — every base upgrade = re-run pipeline + eval + risk sign-off, the $300–500K
   "unbudgeted rework" line. **Sell "migration insurance," not tech.**
3. **Multi-tenant AI app builders** — re-derive N tenant adapters on a new base. Usage-priced.
4. **On-device / edge** — real fit for hypernet-emitted adapters, but long cycles, small budgets. A 2028 story.

### Monetization, by defensibility
- ⭐ **"Migration insurance" enterprise subscription** ($50K–$500K/yr) — priced on risk-avoidance
  and calendar certainty, so it escapes the $0.50/M commodity floor. The anchor revenue.
- ⭐ **Proprietary recipe/curated-data library** subscription ($1K–$20K/mo) — defensible *only*
  if the recipe corpus is proprietary and compounding (ideas #4/#5). Near-zero moat if re-derivable.
- **OEM / white-label** to hosting platforms — fastest distribution, but you're a thin vendor to
  5 buyers who can build it.
- **Per-transfer API** — races the already-cheap cost of just re-running LoRA. Weak.
- **Adapter marketplace take-rate** — ❌ HF proves adapters don't monetize as goods. Skip as primary.

**Realistic size:** single-digit-to-low-double-digit **millions ARR** as a narrow "migration
insurance + eval-trust + proprietary recipe" wedge. Base-case exit is an **acqui-hire**, unless
the compounding recipe flywheel + eval-trust layer becomes something incumbents can't cheaply copy.

### Fastest path to first dollar
Don't build the platform. Land **2–3 enterprise design partners running ≥20 production adapters**,
sell a **paid pilot ($25–75K)** framed against their documented rework cost, timed to the 30 days
after a major base-model drop. Instrument the recipe flywheel from day one — every transfer you
run becomes proprietary corpus. Self-serve API loses to commodity pricing; the paid enterprise
pilot is the fast dollar.

---

## Net recommendation

1. **Lead with Angle B** — portable, compounding training recipes; "curing scientific amnesia."
2. **Run the influence-set-invariance experiment first.** It's the whole thesis's load-bearing
   assumption and a publishable result either way.
3. **Build the trust layer (freshness → verified registry → provenance)** in parallel — it's
   commodity-composition, low research risk, and reuses everything you already built for PromptAura.
4. **Use Angle A only as a downstream demo** (recipe-conditioned emission), never as the headline.
5. **Business framing:** narrow enterprise "migration insurance + eval-trust," not a platform.
   Assume incumbent absorption is the base case; the compounding proprietary corpus is the only
   thing that changes that.

---

## Sources

**Differentiation / novelty**
- [PorTAL — Ramp Labs](https://labs.ramp.com/research/portal-portable-task-adaptation/) · [portallib repo](https://github.com/ramp-public/portallib)
- [Cross-LoRA (arXiv 2508.05232)](https://arxiv.org/abs/2508.05232) · [LoRA-X (arXiv 2501.16559)](https://arxiv.org/pdf/2501.16559)
- [Text-to-LoRA (arXiv 2506.06105)](https://arxiv.org/pdf/2506.06105)
- [Trivial Baselines: Re-evaluating LoRA Adapter Transfer (OpenReview)](https://openreview.net/forum?id=fEeBgr6nlZ)
- [Fine-tuning Transfer / diff-vectors (arXiv 2503.20110)](https://arxiv.org/html/2503.20110v1)
- [LESS — influence-based data selection (Princeton PLI)](https://pli.princeton.edu/blog/2024/using-less-data-tune-models)
- [Metagradient Descent (arXiv 2503.13751)](https://arxiv.org/pdf/2503.13751) · [Curriculum via Beyond Random Sampling (arXiv 2506.11300)](https://arxiv.org/abs/2506.11300)
- [LADDER (arXiv 2503.00735)](https://arxiv.org/pdf/2503.00735) · [Self-Evolving Curriculum (arXiv 2505.14970)](https://arxiv.org/pdf/2505.14970)

**Monetization**
- [Rubrik acquires Predibase ~$100M+ (SiliconANGLE)](https://siliconangle.com/2025/06/25/rubrik-acquires-llm-tooling-startup-predibase-reported-100m/)
- [CoreWeave acquires OpenPipe (TechCrunch)](https://techcrunch.com/2025/09/03/coreweave-acquires-agent-training-startup-openpipe/)
- [Sakana AI $135M Series B / $2.65B (TechCrunch)](https://techcrunch.com/2025/11/17/sakana-ai-raises-135m-series-b-at-a-2-65b-valuation-to-continue-building-ai-models-for-japan)
- [Together AI pricing (eesel)](https://www.eesel.ai/blog/together-ai-pricing) · [Fireworks pricing (UsagePricing)](https://www.usagepricing.com/blueprint/fireworks-ai)
- [AI model refresh cycle breaking enterprise procurement (ValueAdd VC)](https://valueaddvc.com/blog/the-ai-model-refresh-cycle-how-frequent-releases-are-breaking-enterprise-procurement)
- [Hugging Face business breakdown (Contrary Research)](https://research.contrary.com/company/hugging-face)

**New layers**
- [SDFLoRA — federated LoRA (arXiv 2601.11219)](https://arxiv.org/abs/2601.11219)
- [Cross-model task-vector transport (arXiv 2505.12021)](https://arxiv.org/pdf/2505.12021) · [Gradient-sign masking (arXiv 2510.09658)](https://arxiv.org/pdf/2510.09658)
- [Feature-space universality via SAEs (arXiv 2410.06981)](https://arxiv.org/html/2410.06981)
- [Difficulty Is Not Enough — curriculum utility (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/40400)
