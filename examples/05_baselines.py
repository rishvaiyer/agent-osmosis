"""
Example 5 — Baselines: is the curation actually doing anything?

A warm-start speedup is only meaningful if it beats the obvious alternatives. This
compares our coverage recipe against random selection, EL2N-only (hardest examples),
plain distillation, and cold start — all through the same harness, same budget.

    python examples/05_baselines.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osmosis import make_task
from osmosis.baselines import compare_selection

def main():
    task = make_task(n=600, seed=0)
    res = compare_selection(task, budget_frac=0.15, target_acc=0.85, trials=6)

    print(f"Examples to reach {res['target_acc']} accuracy (lower = better):\n")
    print(f"  {'method':<20} {'examples':>10}")
    for r in res["rows"]:
        val = f"{r['examples_to_target']:.0f} ± {r['ci']:.0f}" if r["examples_to_target"] else "never"
        star = "  ⭐" if r["method"] == "coverage (ours)" else ""
        print(f"  {r['method']:<20} {val:>10}{star}")
    if res["ours_vs_cold"]:
        print(f"\n  ours vs cold: {res['ours_vs_cold']:.2f}x fewer examples")

    rows = [r for r in res["rows"] if r["examples_to_target"]]
    names = [r["method"] for r in rows]
    vals = [r["examples_to_target"] for r in rows]
    colors = ["#0E8F8A" if n == "coverage (ours)" else "#7A929C" for n in names]
    plt.figure(figsize=(7.5, 4.2))
    plt.barh(names, vals, color=colors)
    plt.xlabel("labeled examples to target (lower is better)")
    plt.title("agent-osmosis — coverage recipe vs the obvious baselines")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("examples/out_baselines.png", dpi=130)
    print("  saved examples/out_baselines.png")


if __name__ == "__main__":
    main()
