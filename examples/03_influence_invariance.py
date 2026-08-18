"""
Example 3 — The load-bearing experiment.

Do different base models agree on which examples are most influential? If yes, the
expensive "which data matters" computation can be done once and transferred — the
premise the whole recipe idea rests on. We compute the top influential set for four
different base models and measure how much they overlap (vs a random baseline).

    python examples/03_influence_invariance.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from osmosis import make_task, influence_invariance

def main():
    task = make_task(n=600, seed=0)
    inv = influence_invariance(task)

    print(f"Top-{inv['k']} influential examples, agreement across {len(inv['bases'])} base models:")
    print("           " + "  ".join(b.split('-')[-1] for b in inv["bases"]))
    for b, row in zip(inv["bases"], inv["matrix"]):
        print(f"  {b:<12} " + "     ".join(f"{v:.2f}" for v in row))
    print(f"\n  mean cross-model overlap : {inv['mean_overlap']:.3f}")
    print(f"  random baseline          : {inv['random_overlap']:.3f}")
    print(f"  verdict                  : {inv['verdict'].upper()} "
          f"({inv['mean_overlap']/max(inv['random_overlap'],1e-6):.1f}x above chance)")

    m = np.array(inv["matrix"])
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(m, cmap="BuGn", vmin=0, vmax=1)
    labels = [b.split("-")[-1] for b in inv["bases"]]
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                     color="#15242b" if m[i, j] < 0.6 else "white", fontsize=10)
    plt.colorbar(label="Jaccard overlap of top-influential sets")
    plt.title(f"Influence agreement across base models\n(random baseline ≈ {inv['random_overlap']:.2f})")
    plt.tight_layout()
    out = "examples/out_invariance.png"
    plt.savefig(out, dpi=130)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
