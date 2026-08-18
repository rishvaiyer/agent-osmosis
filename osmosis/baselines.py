"""
Baselines — the honest comparison.

A warm-start speedup only means something if it beats the obvious alternatives. This
module pits the coverage recipe against the baselines a skeptic would ask about:

  cold          : no recipe at all (train from scratch)
  random        : a random class-balanced subset of the same size (is curation real?)
  el2n          : top-k by pure informativeness / EL2N (no difficulty-coverage)
  distillation  : the source model pseudo-labels the data, target imitates (the
                  cheap, strong baseline every transfer method must beat)
  coverage      : ours — class-balanced, margin-stratified coverage coreset

Metric: unique labeled examples to reach the target accuracy (lower is better).
"""
from __future__ import annotations
import numpy as np

from .data import Task
from .models import OsmosisModel, Curve
from .recipe import Recipe
from .transfer import warm_start, cold_start, extract_recipe
from .metrics import mean_ci


def _recipe_from_indices(task: Task, idx, source_base: str) -> Recipe:
    """Wrap a chosen set of example indices as a minimal Recipe so warm_start can run it."""
    idx = [int(i) for i in idx]
    return Recipe(task=task.name, source_model=source_base,
                  influence_set=idx, ordering=idx, method="baseline")


def _random_coreset(task: Task, k: int, seed: int) -> np.ndarray:
    """A class-balanced random subset — the 'is curation actually doing anything?' control."""
    tr_texts, tr_y = task.slice(task.train_idx)
    tr_y = np.asarray(tr_y)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(tr_y):
        ci = np.where(tr_y == c)[0]
        chosen += list(rng.choice(ci, min(k // 2, len(ci)), replace=False))
    return np.array(chosen[:k])


def _el2n_coreset(task: Task, k: int, source_base: str) -> np.ndarray:
    """Top-k by pure informativeness (EL2N-style) — no difficulty coverage. Tests
    whether spreading across difficulty (our trick) beats just taking the hardest."""
    tr_texts, tr_y = task.slice(task.train_idx)
    tr_y = np.asarray(tr_y)
    m = OsmosisModel(base_model=source_base, seed=0)
    for _ in range(3):
        m._partial_fit(tr_texts, tr_y)
    loss = m.per_example_loss(tr_texts, tr_y)     # high loss = "hard"/informative
    return np.argsort(-loss)[:k]


def _distillation_cost(task: Task, source_base: str, target_base: str,
                       target_acc: float, hp: dict, seed: int) -> int | None:
    """Distillation baseline: the source labels every example, the target learns from
    the source's labels (imitation). Measured on TRUE val labels, in unique examples."""
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)
    src = OsmosisModel(base_model=source_base, lr=hp["lr"], seed=0)
    src.train_stream(tr_texts, tr_y, val_texts, val_y, batch=hp["batch"])
    pseudo = src.predict(tr_texts)                # the teacher's labels
    student = OsmosisModel(base_model=target_base, lr=hp["lr"], seed=seed)
    rng = np.random.default_rng(seed); order = rng.permutation(len(tr_texts))
    curve = student.train_stream([tr_texts[i] for i in order], pseudo[order],
                                 val_texts, val_y, batch=hp["batch"])
    return curve.steps_to_target(target_acc)


def compare_selection(task: Task, source_base="osmo-base-a", target_base="osmo-base-b",
                      budget_frac=0.15, target_acc=0.85, trials=6, hp=None):
    """Run every method through the same warm-vs-cold harness and return a comparison
    table: mean examples-to-target (± CI) for each. Ours should win; the point is to
    prove it beats random, EL2N-only, and distillation — not just cold start."""
    hp = hp or {"lr": 0.15, "batch": 16, "prefit_passes": 3}
    tr_texts, _ = task.slice(task.train_idx)
    k = max(4, int(len(tr_texts) * budget_frac))

    coverage_recipe = extract_recipe(task, source_base=source_base, budget_frac=budget_frac, hp=hp)
    methods = {
        "coverage (ours)": coverage_recipe,
        "random":  _recipe_from_indices(task, _random_coreset(task, k, 0), source_base),
        "el2n":    _recipe_from_indices(task, _el2n_coreset(task, k, source_base), source_base),
    }

    rows = []
    for name, recipe in methods.items():
        steps = [warm_start(task, recipe, target_base, target_acc, hp, seed=s)[0]
                 .steps_to_target(target_acc) for s in range(trials)]
        m, ci = mean_ci(steps)
        rows.append({"method": name, "examples_to_target": m, "ci": ci, "coreset": k})

    # cold + distillation (no coreset)
    cold = [cold_start(task, target_base, target_acc, hp, seed=s).steps_to_target(target_acc)
            for s in range(trials)]
    cm, cci = mean_ci(cold)
    rows.append({"method": "cold (no recipe)", "examples_to_target": cm, "ci": cci, "coreset": 0})
    dist = [_distillation_cost(task, source_base, target_base, target_acc, hp, s) for s in range(trials)]
    dm, dci = mean_ci(dist)
    rows.append({"method": "distillation", "examples_to_target": dm, "ci": dci, "coreset": len(tr_texts)})

    # rank (lower examples = better); None (never reached target) sinks to the bottom
    rows.sort(key=lambda r: (r["examples_to_target"] is None, r["examples_to_target"] or 1e9))
    best = next((r for r in rows if r["method"] == "coverage (ours)"), None)
    cold_row = next((r for r in rows if r["method"] == "cold (no recipe)"), None)
    return {"rows": rows, "target_acc": target_acc,
            "ours_vs_cold": (cold_row["examples_to_target"] / best["examples_to_target"])
            if (best and cold_row and best["examples_to_target"]) else None}
