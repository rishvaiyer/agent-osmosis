"""
Example 2 — The recipe compounds.

Successive model 'generations' each run the recipe and fold back the failures they
still had. The recipe gets better (accuracy from the recipe alone climbs) and
cheaper to reach target with, generation over generation. This compounding is the
part a one-shot adapter transfer can't do.

    python examples/02_compounding.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osmosis import compounding_chain

TARGET = 0.82

def main():
    ch = compounding_chain(target_acc=TARGET, trials=5)
    gens = ch["generations"]

    print("gen  base           recipe-quality  warm-cost  size  confidence")
    for g in gens:
        wm = f"{g['warm_mean']:.0f}" if g["warm_mean"] else "miss"
        print(f" {g['generation']}   {g['base']:<13}  {g['recipe_quality']:.3f}          "
              f"{wm:<9}  {g['recipe_size']:<4}  {g['confidence']:.0f}")

    xs = [g["generation"] for g in gens]
    quality = [g["recipe_quality"] for g in gens]
    warm = [g["warm_mean"] for g in gens]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(xs, quality, "-o", color="#0E8F8A", lw=2.5, label="recipe quality (acc from recipe alone)")
    ax1.set_xlabel("model generation")
    ax1.set_ylabel("recipe quality", color="#0E8F8A")
    ax1.set_xticks(xs)
    ax2 = ax1.twinx()
    ax2.plot(xs, warm, "-s", color="#D9612E", lw=2.2, label="warm cost (examples to target)")
    ax2.set_ylabel("labeled examples to target", color="#D9612E")
    fig.suptitle("agent-osmosis — the recipe compounds across generations")
    fig.tight_layout()
    out = "examples/out_compounding.png"
    plt.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
