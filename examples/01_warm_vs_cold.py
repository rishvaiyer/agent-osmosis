"""
Example 1 — Warm start vs cold start.

Extract a recipe from model A, use it to warm-start a *different* model B, and
show B reaches the target accuracy using far fewer unique labeled examples than
training from scratch.

    python examples/01_warm_vs_cold.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osmosis import make_task, warm_vs_cold

TARGET = 0.85

def main():
    task = make_task(n=600, seed=0)
    r = warm_vs_cold(task, source_base="osmo-base-a", target_base="osmo-base-b",
                     budget_frac=0.15, target_acc=TARGET, trials=8)

    print(f"Task: {task.name}   target accuracy: {TARGET}")
    print(f"  cold start : {r['cold_mean']:.0f} ± {r['cold_ci']:.0f} labeled examples")
    print(f"  warm start : {r['warm_mean']:.0f} ± {r['warm_ci']:.0f} labeled examples")
    print(f"  speedup    : {r['speedup']:.2f}x  (recipe = {len(r['recipe'].influence_set)} examples,"
          f" transfer-confidence {r['recipe'].transfer_confidence():.0f}/100)")

    plt.figure(figsize=(7, 4.2))
    plt.plot(r["grid"], r["cold_curve"], label="cold start (from scratch)", lw=2.2, color="#7A929C")
    plt.plot(r["grid"], r["warm_curve"], label="warm start (recipe)", lw=2.6, color="#0E8F8A")
    plt.axhline(TARGET, ls="--", color="#D9612E", lw=1.3, label=f"target {TARGET}")
    plt.xlabel("unique labeled examples seen")
    plt.ylabel("validation accuracy")
    plt.title("agent-osmosis — warm start reaches target with less data")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = "examples/out_warm_vs_cold.png"
    plt.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
