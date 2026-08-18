"""
Smoke + property tests for the agent-osmosis engine. Run: `pytest -q`.

These assert the three claims the MVP makes actually hold on the demo substrate:
warm beats cold, influence is invariant across models, and the recipe compounds.
"""
import numpy as np
from osmosis import (make_task, warm_vs_cold, influence_invariance,
                     compounding_chain, extract_recipe, Recipe)


def test_task_is_learnable():
    task = make_task(n=400, seed=1)
    assert task.n == 400
    assert set(np.unique(task.labels)) <= {0, 1}


def test_warm_beats_cold():
    task = make_task(n=500, seed=0)
    r = warm_vs_cold(task, trials=6, target_acc=0.85)
    assert r["warm_mean"] is not None and r["cold_mean"] is not None
    # the whole point: warm reaches target with fewer labeled examples
    assert r["warm_mean"] < r["cold_mean"]
    assert r["speedup"] > 1.0


def test_influence_is_invariant():
    task = make_task(n=500, seed=0)
    inv = influence_invariance(task)
    # cross-model agreement must be clearly above the random baseline
    assert inv["mean_overlap"] > 3 * inv["random_overlap"]
    assert inv["verdict"] == "invariant"


def test_recipe_compounds():
    ch = compounding_chain(trials=4, target_acc=0.82)
    q = [g["recipe_quality"] for g in ch["generations"]]
    # quality should trend upward across generations
    assert q[-1] > q[0]


def test_recipe_roundtrip(tmp_path):
    task = make_task(n=300, seed=2)
    recipe = extract_recipe(task, budget_frac=0.15)
    p = tmp_path / "r.json"
    recipe.save(str(p))
    loaded = Recipe.load(str(p))
    assert loaded.influence_set == recipe.influence_set
    assert loaded.task == recipe.task


def test_freshness_states():
    task = make_task(n=300, seed=2)
    recipe = extract_recipe(task, budget_frac=0.15)
    assert recipe.freshness() == "unverified"  # no transfers recorded yet
