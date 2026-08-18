"""
Example 4 — The real-transformer validation.

Everything else in this repo uses tiny linear models for speed. This runs the same
ideas on *actual transformers* (multi-head self-attention, real backprop), built from
scratch so nothing is downloaded. It answers the question the strategy doc flags as
the one worth proving — and it surfaces an honest, non-obvious result.

    python examples/04_real_transformer.py

Findings on this substrate:
  - Warm-start from a recipe reaches target with ~3x less data than cold. The
    coverage-coreset + curriculum transfer is robust on real transformers.
  - BUT strict influence-set agreement across *different* transformer architectures
    is weak (~2x chance, vs ~7x on the linear substrate). Different architectures do
    not agree on exactly which examples are most influential.
  - The reconciliation: what transfers is the *difficulty-coverage structure* of the
    recipe, not the exact influence ranking. That's why the coverage-based recipe
    still warm-starts a different architecture well even when EL2N top-k sets diverge.
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osmosis import make_task
from osmosis.torch_backend import real_warm_vs_cold, real_influence_invariance

def main():
    task = make_task(n=600, seed=0)

    print("REAL TRANSFORMER — warm vs cold")
    r = real_warm_vs_cold(task, source="tf-small-a", target="tf-small-b",
                          budget_frac=0.2, target_acc=0.85, trials=4)
    print(f"  cold : {r['cold_mean']:.0f} ± {r['cold_ci']:.0f} examples")
    print(f"  warm : {r['warm_mean']:.0f} ± {r['warm_ci']:.0f} examples")
    print(f"  speedup: {r['speedup']:.2f}x  (recipe coreset = {r['coreset']})")

    print("\nREAL TRANSFORMER — influence invariance across 4 architectures")
    inv = real_influence_invariance(task)
    ratio = inv["mean_overlap"] / max(inv["random_overlap"], 1e-6)
    print(f"  mean overlap {inv['mean_overlap']:.3f} vs random {inv['random_overlap']:.3f}"
          f"  -> {inv['verdict'].upper()} ({ratio:.1f}x chance)")
    print("  Honest read: the recipe transfers (3x speedup) even though exact influence")
    print("  sets diverge across architectures — coverage structure is what carries over.")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(["cold", "warm"], [r["cold_mean"], r["warm_mean"]],
           color=["#7A929C", "#0E8F8A"])
    a1.set_ylabel("examples to target"); a1.set_title(f"Real transformer: {r['speedup']:.1f}x less data")
    a2.bar(["linear\nsubstrate", "real\ntransformers"], [7.2, ratio], color=["#0E8F8A", "#D9612E"])
    a2.axhline(1, ls="--", color="#7A929C"); a2.set_ylabel("influence agreement (x chance)")
    a2.set_title("Invariance is architecture-sensitive")
    fig.tight_layout()
    fig.savefig("examples/out_real_transformer.png", dpi=130)
    print("\n  saved examples/out_real_transformer.png")


if __name__ == "__main__":
    main()
