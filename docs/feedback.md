# Candid Feedback — model-osmosis

*No cheerleading. What I'd tell a friend before they spent three months on this.*

## Scores

| Lens | Score | One-liner |
| --- | --- | --- |
| Instinct / taste in problems | **8/10** | You independently walked to real frontiers. Good nose. |
| Concept novelty (as first framed) | **4/10** | The frontiers you found are already occupied — Sakana, Cross-LoRA, **PorTAL**. |
| Angle A (unseen-base hypernetwork) | **3/10** | **Shipped by Ramp Labs in July 2026.** As-framed, you're a fast-follower. |
| Angle B (learning-trajectory / recipe transfer) | **7/10** | Genuinely under-occupied. The one real opening. |
| As a research contribution | **5/10** | One publishable result (influence-set invariance) is in reach; the grand claim is a stretch. |
| As a business / moat | **3/10** | Commodity floor + incumbent absorption. A wedge, not a platform. |
| As a build-and-learn project | **9/10** | Ship the warm-vs-cold graph and you look like someone who executes. |

## The three things you most need to hear

**1. Angle A is gone. Accept it fast.** Ramp Labs' PorTAL already does "emit an adapter for an
unseen base" at ~98% of per-task-LoRA quality. I told you earlier this was open whitespace — the
research corrected me. Don't spend a day trying to out-PorTAL PorTAL. Use adapter-emission only as
a *downstream demo of the recipe idea* (condition emission on the transferred recipe, not on task
text) — that's the one framing that's still distinct.

**2. Distillation is the silent idea-killer.** The deepest problem with the whole concept: if a
teacher model already learned the task, plain distillation into a student is cheap and near-optimal.
Every fancier thing you build has to prove it beats plain distillation. Often the endpoint carries
all the signal the path did. Budget an honest ablation against "just distill it" and against "just
replay the same data in the same order." If dumb baselines win — and the *Trivial Baselines* paper
says they often do — the machinery is over-engineered.

**3. Your real edge isn't the ML. It's the trust layer — and you've already built it once.** The
most defensible, least-crowded pieces (freshness tracking, verified registry with reputation,
provenance receipts) are *exactly PromptAura's mechanics one layer down at the weights.* That's not
a coincidence to paper over — it's the actual insight. "Verified to transfer to model X, on date Z,
by a creator whose adapters survive re-verification" is a thing nobody offers and you're uniquely
positioned to build. Lead with that identity, not with hypernetwork math you'll lose on.

## What would make me change the scores upward

- **Run the influence-set-invariance experiment** and show top-k influential examples correlate
  across heterogeneous bases. Small, clean, publishable, and it's the load-bearing assumption of the
  entire recipe thesis. This single result moves "concept novelty" from 4 to 6.
- **Produce the money graph**: cost-to-target-accuracy falling monotonically across a chain of 3–4
  successive base models. If that curve is real and not flat after gen 2, the "compounding" claim is
  true and this becomes genuinely impressive. If it's flat, you learned that cheaply — retreat to
  the one-shot claim.
- **Land one paying design partner** on "migration insurance." Revenue changes every conversation.

## The honest bottom line

The idea, as a *company*, is a 3–4: real pain, real buyers, but structurally a feature that Together
/ Fireworks / CoreWeave will absorb, on tech Sakana is open-sourcing. As a *research narrative*, it's
a 5 that could become a 7 with one good experiment. As a *portfolio piece that proves you can take a
shower-thought to a real speedup graph*, it's a 9 — and that's the version most worth doing.

Do this: **build the Angle-B MVP, run the invariance experiment, ship one graph.** Decide whether
it's a company *after* you have the number — not before.
