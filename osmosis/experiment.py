"""
The three experiments that make the case.

1. warm_vs_cold        — does a recipe get model B to target accuracy in fewer examples?
2. compounding_chain   — does the recipe get *cheaper* across successive model generations?
3. influence_invariance— do different base models agree on which examples matter?
                         (the load-bearing assumption of the whole idea)
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from .data import Task, make_task, make_task_family
from .models import OsmosisModel, BASE_MODELS
from .influence import top_influential
from .recipe import Recipe
from .transfer import extract_recipe, cold_start, warm_start, discover_fixes
from .metrics import jaccard, mean_ci, interp_curve

DEFAULT_HP = {"lr": 0.15, "batch": 16, "prefit_passes": 3}


def warm_vs_cold(task: Task, source_base="osmo-base-a", target_base="osmo-base-b",
                 budget_frac=0.15, target_acc=0.85, trials=8, hp=None):
    """Run many seeded trials; return averaged curves + steps-to-target with CIs."""
    hp = hp or DEFAULT_HP
    recipe = extract_recipe(task, source_base=source_base, budget_frac=budget_frac, hp=hp)

    warm_steps, cold_steps = [], []
    warm_curves, cold_curves = [], []
    for s in range(trials):
        cc = cold_start(task, target_base, target_acc, hp, seed=s)
        wc, receipt = warm_start(task, recipe, target_base, target_acc, hp, seed=s)
        cold_steps.append(cc.steps_to_target(target_acc))
        warm_steps.append(wc.steps_to_target(target_acc))
        cold_curves.append(cc); warm_curves.append(wc)
        recipe.record_transfer(receipt)

    # average the curves onto a shared x-grid
    max_x = max(max(c.seen) for c in cold_curves + warm_curves)
    grid = np.linspace(0, max_x, 60)
    warm_avg = np.mean([interp_curve(c.seen, c.val_acc, grid) for c in warm_curves], axis=0)
    cold_avg = np.mean([interp_curve(c.seen, c.val_acc, grid) for c in cold_curves], axis=0)

    cm, ch = mean_ci(cold_steps)
    wm, wh = mean_ci(warm_steps)
    speedup = (cm / wm) if (cm and wm) else None
    return {
        "recipe": recipe,
        "grid": grid.tolist(),
        "warm_curve": warm_avg.tolist(), "cold_curve": cold_avg.tolist(),
        "warm_steps": warm_steps, "cold_steps": cold_steps,
        "warm_mean": wm, "warm_ci": wh, "cold_mean": cm, "cold_ci": ch,
        "speedup": speedup, "target_acc": target_acc,
    }


def _recipe_quality(task: Task, recipe: Recipe, base: str, hp: dict, trials: int = 4) -> float:
    """Accuracy a fresh model reaches from the recipe *alone* (no extra streaming).
    This is the cleanest measure of how good the recipe is — and it should climb as
    the recipe compounds."""
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)
    order = [i for i in recipe.ordering if i < len(tr_texts)]
    accs = []
    for s in range(trials):
        m = OsmosisModel(base_model=base, lr=hp["lr"], seed=s)
        m.warm_prefit([tr_texts[i] for i in order], tr_y[order], passes=hp.get("prefit_passes", 3))
        accs.append(m.accuracy(val_texts, val_y))
    return float(np.mean(accs))


def compounding_chain(bases=("osmo-base-a", "osmo-base-b", "osmo-base-c", "osmo-wide-d"),
                      target_acc=0.85, start_budget_frac=0.05, trials=5, hp=None, seed=0):
    """One task, successive model 'generations'. The recipe starts small and each
    generation folds back the failures *it* discovered. We track recipe quality
    (accuracy from the recipe alone) and warm cost as the recipe compounds — the
    recipe should get better, and cheaper to reach target with, generation over generation."""
    hp = hp or DEFAULT_HP
    task = make_task(name="shared-task", n=700, seed=seed)

    # deliberately weak seed recipe, so there is room to improve
    recipe = extract_recipe(task, source_base=bases[0], budget_frac=start_budget_frac, hp=hp)
    generations = []
    for gen, base in enumerate(bases):
        quality = _recipe_quality(task, recipe, base, hp)
        warm = [warm_start(task, recipe, base, target_acc, hp, seed=s) for s in range(trials)]
        for _, rc in warm:
            recipe.record_transfer(rc)
        wm, wh = mean_ci([c.steps_to_target(target_acc) for c, _ in warm])
        cm, chw = mean_ci([cold_start(task, base, target_acc, hp, seed=s).steps_to_target(target_acc)
                           for s in range(trials)])
        generations.append({
            "generation": gen + 1, "base": base,
            "recipe_quality": round(quality, 3),
            "warm_mean": wm, "warm_ci": wh, "cold_mean": cm, "cold_ci": chw,
            "recipe_version": recipe.version, "recipe_size": len(recipe.influence_set),
            "failures_logged": len(recipe.failure_fix_log),
            "confidence": recipe.transfer_confidence(),
        })
        # compound: fold in the failures THIS base still had
        new_fixes = discover_fixes(task, base, recipe, hp)
        recipe.compound(new_fixes, stamp=datetime.now(timezone.utc).isoformat())
    return {"generations": generations, "recipe": recipe, "target_acc": target_acc, "task": task}


def influence_invariance(task: Task, bases=None, budget_frac=0.15, hp=None):
    """THE de-risking experiment: do different base models pick the same influential
    examples? Returns the top-k set per base and the pairwise Jaccard overlap matrix.
    Also reports overlap vs a random baseline for calibration."""
    hp = hp or DEFAULT_HP
    bases = list(bases or BASE_MODELS.keys())
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)
    k = max(4, int(len(tr_texts) * budget_frac))

    from .influence import influence_scores
    top_sets = {}
    for b in bases:
        m = OsmosisModel(base_model=b, lr=hp["lr"], seed=0)
        m.train_stream(tr_texts, tr_y, val_texts, val_y, batch=hp["batch"])
        # "which examples are most influential" = top-k by informativeness score.
        scores = influence_scores(m, tr_texts, tr_y)
        top_sets[b] = set(int(i) for i in np.argsort(-scores)[:k])

    matrix = [[round(jaccard(top_sets[a], top_sets[b]), 3) for b in bases] for a in bases]
    # random baseline: expected Jaccard of two random k-subsets of n
    n = len(tr_texts)
    rng = np.random.default_rng(0)
    rand = [jaccard(rng.choice(n, k, replace=False), rng.choice(n, k, replace=False))
            for _ in range(20)]
    off = [matrix[i][j] for i in range(len(bases)) for j in range(len(bases)) if i != j]
    return {
        "bases": bases, "k": k, "matrix": matrix,
        "mean_overlap": round(float(np.mean(off)), 3),
        "random_overlap": round(float(np.mean(rand)), 3),
        "verdict": "invariant" if np.mean(off) > 3 * np.mean(rand) else "weakly-invariant",
    }
