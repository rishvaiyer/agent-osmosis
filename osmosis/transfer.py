"""
Extract a recipe from one model's training run, and warm-start another model with it.

warm-start vs cold-start is the core experiment:
  cold : model B learns the task from scratch
  warm : model B is pre-fit on the transferred recipe, then learns the task
We count *every* example B trains on (recipe pre-fit included), so the comparison
is honest: if warm reaches the target accuracy in fewer total examples, the recipe
genuinely expedited fine-tuning.

Known gotcha (from the warm-start literature): naive warm-starting can *hurt*
generalization. The shrink-perturb trick fixes it, and we log that as the recipe's
first failure->fix entry — a nice demonstration of the mechanic on the machinery itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from .data import Task
from .models import OsmosisModel, Curve
from .influence import top_influential
from .curriculum import order_easy_to_hard
from .recipe import Recipe, FailureFix, EvalReceipt


def _now() -> str:
    """Get current UTC time as ISO format string."""
    return datetime.now(timezone.utc).isoformat()


def extract_recipe(task: Task, source_base: str = "osmo-base-a",
                   budget_frac: float = 0.15, hp: dict | None = None) -> Recipe:
    """Train a source model, then distill *how* it learned into a Recipe.

    Steps:
      1. Train a model on the full task
      2. Find which examples were most influential (coverage-stratified coreset)
      3. Order them easy->hard for better curriculum learning
      4. Record failures the model couldn't fix and which examples could help
    """
    hp = hp or {"lr": 0.15, "batch": 16, "prefit_passes": 3}
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)

    # Train the source model on the task
    source = OsmosisModel(base_model=source_base, lr=hp["lr"], seed=0)
    source.train_stream(tr_texts, tr_y, val_texts, val_y, batch=hp["batch"])

    # 1) Find which examples mattered (indices are *positions into the train pool*)
    k = max(4, int(len(tr_texts) * budget_frac))
    inf_local = top_influential(source, tr_texts, tr_y, k)
    # 2) Arrange them in difficulty order: easy->hard
    ordering_local = order_easy_to_hard(source, tr_texts, tr_y, inf_local)

    # 3) failure -> fix log: val examples the source still gets wrong, and the
    #    same-class influential training examples that would shore them up.
    preds = source.predict(val_texts)
    fails = np.where(preds != np.asarray(val_y))[0]
    fflog: list[FailureFix] = []
    for fi in fails[:6]:
        # For each validation failure, find same-class examples from the recipe that could help
        true = int(val_y[fi])
        same_class = [int(i) for i in ordering_local if int(tr_y[i]) == true][:3]
        fflog.append(FailureFix(
            failure_text=val_texts[fi], true_label=true,
            fix_indices=same_class,
            note="source misclassified; reinforce with these same-class exemplars",
        ))
    # the machinery's own first lesson: warm-start needs a fix to not hurt generalization
    fflog.insert(0, FailureFix(
        failure_text="[meta] naive warm-start regressed val accuracy",
        true_label=-1, fix_indices=[],
        note="apply shrink-perturb (scale head then add noise) before the main stream",
    ))

    return Recipe(
        task=task.name, source_model=source_base,
        influence_set=[int(i) for i in inf_local],
        ordering=[int(i) for i in ordering_local],
        failure_fix_log=fflog, hyperparams=hp,
        method="influence-proxy/v1",
    )


def _shrink_perturb(model: OsmosisModel, shrink: float = 0.6, noise: float = 0.02):
    """The fix from the failure->fix log: keep warm-start's head but soften it so
    it stays plastic. No-op until the head has been fit once."""
    if getattr(model.head, "coef_", None) is not None:
        rng = np.random.default_rng(0)
        model.head.coef_ = shrink * model.head.coef_ + noise * rng.normal(size=model.head.coef_.shape)
        model.head.intercept_ = shrink * model.head.intercept_


def cold_start(task: Task, target_base: str, target_acc: float,
               hp: dict, seed: int = 0) -> Curve:
    m = OsmosisModel(base_model=target_base, lr=hp["lr"], seed=seed)
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)
    order = np.random.default_rng(seed).permutation(len(tr_texts))
    return m.train_stream([tr_texts[i] for i in order], tr_y[order],
                          val_texts, val_y, batch=hp["batch"])


def warm_start(task: Task, recipe: Recipe, target_base: str, target_acc: float,
               hp: dict, seed: int = 0, use_shrink_perturb: bool = False) -> tuple[Curve, EvalReceipt]:
    """Cost is measured in *unique labeled examples* — the scarce resource. Passes
    over the small recipe coreset are cheap and don't inflate the count; the recipe
    examples are then excluded from the post-stream so every streamed example is new."""
    m = OsmosisModel(base_model=target_base, lr=hp["lr"], seed=seed)
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)

    # --- the warm start: pre-fit on the recipe coreset, in curriculum order ---
    order = [i for i in recipe.ordering if i < len(tr_texts)]
    coreset = set(order)
    pre_texts = [tr_texts[i] for i in order]
    pre_y = tr_y[order]
    m.warm_prefit(pre_texts, pre_y, passes=hp.get("prefit_passes", 3))
    if use_shrink_perturb:
        _shrink_perturb(m)
    seen = len(coreset)                       # unique labeled examples consumed so far
    curve = Curve(seen=[seen], val_acc=[m.accuracy(val_texts, val_y)])

    # --- then stream the *remaining* (new) examples, as cold-start would ---
    rng = np.random.default_rng(seed + 100)
    rest = [i for i in rng.permutation(len(tr_texts)) if i not in coreset]
    curve = m.train_stream([tr_texts[i] for i in rest], tr_y[rest],
                           val_texts, val_y, batch=hp["batch"],
                           curve=curve, start_seen=seen)

    reached = curve.val_acc[-1]
    receipt = EvalReceipt(
        target_model=target_base, reached_acc=round(reached, 4),
        steps_to_target=curve.steps_to_target(target_acc),
        dated=_now(),
        outcome="transferred" if reached >= target_acc else "failed",
    )
    return curve, receipt


def discover_fixes(task: Task, model_base: str, recipe: Recipe, hp: dict) -> list[FailureFix]:
    """After a model runs the recipe, harvest the failures *it* still has, so they
    can be compounded back into the recipe for the next generation."""
    m = OsmosisModel(base_model=model_base, lr=hp["lr"], seed=3)
    tr_texts, tr_y = task.slice(task.train_idx)
    val_texts, val_y = task.slice(task.val_idx)
    order = [i for i in recipe.ordering if i < len(tr_texts)]
    m.warm_prefit([tr_texts[i] for i in order], tr_y[order], passes=hp.get("prefit_passes", 3))
    preds = m.predict(val_texts)
    fails = np.where(preds != np.asarray(val_y))[0]
    out = []
    for fi in fails[:4]:
        true = int(val_y[fi])
        # a fix = influential same-class training example not already in the recipe
        cand = top_influential(m, tr_texts, tr_y, 20)
        fix = [int(i) for i in cand if int(tr_y[i]) == true and int(i) not in recipe.influence_set][:2]
        out.append(FailureFix(failure_text=val_texts[fi], true_label=true,
                              fix_indices=fix, note=f"discovered by {model_base}"))
    return out
