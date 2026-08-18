"""
The Recipe — a portable, model-agnostic record of *how* a task was learned.

This is the whole idea. Instead of shipping trained weights (which don't transfer
between architectures), we ship the training *program*:

  - influence_set   : the small subset of examples that carried the signal
  - ordering        : the curriculum (what order to teach them in)
  - failure_fix_log : discovered failure modes -> the examples that fix them
  - hyperparams     : the dials that worked
  - eval_receipts   : dated proof it reached score X on model Y  (trust layer)

A Recipe can be `compound`ed: each model that runs it appends what it learned, so
the recipe gets better every generation. The freshness function is ported directly
from promptaura's `freshnessOf` — the same trust mechanic, one layer down at the weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json

FRESH_DAYS = 30
STALE_DAYS = 90


@dataclass
class FailureFix:
    """Records a model mistake and the training examples that could fix it.

    When a model misclassifies a validation example, we log it here along with
    similar (correct) training examples from the same class that would teach the
    model not to make that mistake again.
    """
    failure_text: str          # a validation example the source model got wrong
    true_label: int
    fix_indices: list[int]     # training examples that, added, fix this failure
    note: str = ""


@dataclass
class EvalReceipt:
    """Proof that a recipe worked (or didn't) on a target model at a specific time.

    When we test a recipe on a new model, we record whether it reached our accuracy
    target and how many examples it took. This timestamped record builds up a
    history of the recipe's performance across models.
    """
    target_model: str
    reached_acc: float
    steps_to_target: int | None
    dated: str                 # ISO timestamp
    outcome: str               # "transferred" | "failed"


@dataclass
class Recipe:
    """A portable training program: which examples to use, in what order, and how fixes compound.

    Instead of shipping trained weights (which don't transfer between architectures),
    we ship the recipe — the data, the order, and a history of what worked. A new model
    can follow this recipe to learn the task, and fold back its own mistakes so the
    recipe gets better for the next model.
    """
    task: str
    source_model: str
    influence_set: list[int]                       # indices into the task's training pool
    ordering: list[int]                            # same indices, curriculum order
    failure_fix_log: list[FailureFix] = field(default_factory=list)
    hyperparams: dict = field(default_factory=dict)
    eval_receipts: list[EvalReceipt] = field(default_factory=list)
    method: str = "influence-proxy/v1"
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # -- trust layer (ported from promptaura freshnessOf) -------------------
    def freshness(self, now: datetime | None = None) -> str:
        """Rate the recipe's trustworthiness based on success history and age.

        A recipe is 'fresh' if it recently worked. It becomes 'stale' if not tested
        recently, 'edited' if changed after the last successful test, or 'broken' if
        it failed more often than it worked.
        """
        now = now or datetime.now(timezone.utc)
        # Count successes and failures in eval_receipts
        worked = sum(1 for r in self.eval_receipts if r.outcome == "transferred")
        failed = sum(1 for r in self.eval_receipts if r.outcome == "failed")
        # If failures outnumber successes, the recipe is broken
        if failed > 0 and failed >= worked:
            return "broken"
        # If never tested, it's unverified
        if worked == 0:
            return "unverified"
        # Find the most recent successful transfer
        last = max((datetime.fromisoformat(r.dated) for r in self.eval_receipts
                    if r.outcome == "transferred"), default=None)
        if last is None:
            return "unverified"
        # If edited after last success, can't trust it based on old results
        if datetime.fromisoformat(self.updated_at) > last:
            return "edited"
        # Otherwise, rate by age: fresh (0-30 days), aging (30-90), or stale (90+)
        days = (now - last).total_seconds() / 86_400
        if days <= FRESH_DAYS:
            return "fresh"
        if days <= STALE_DAYS:
            return "aging"
        return "stale"

    def transfer_confidence(self) -> float:
        """A 0-100 score for the dashboard gauge: how much to trust this recipe.

        Combines four signals: success rate, cross-model breadth, recency, and
        how much the recipe has improved over time (compounding).
        """
        worked = [r for r in self.eval_receipts if r.outcome == "transferred"]
        failed = [r for r in self.eval_receipts if r.outcome == "failed"]
        if not self.eval_receipts:
            return 0.0
        # Success rate: fraction of tries that worked
        success = len(worked) / len(self.eval_receipts)
        # Breadth: how many different model architectures worked (up to 3 models)
        breadth = min(len({r.target_model for r in worked}) / 3.0, 1.0)  # cross-model breadth
        # Recency: weight recent successes higher than old ones
        recency = {"fresh": 1.0, "aging": 0.6, "stale": 0.3,
                   "edited": 0.4, "broken": 0.0, "unverified": 0.2}[self.freshness()]
        # Compounding: has the recipe gotten better by folding in discovered fixes
        compounding = min(len(self.failure_fix_log) / 10.0, 1.0)
        # Weighted average: prioritize success, recency, breadth, then compounding
        return round(100 * (0.4 * success + 0.25 * breadth + 0.25 * recency + 0.10 * compounding), 1)

    # -- compounding --------------------------------------------------------
    def record_transfer(self, receipt: EvalReceipt):
        """Log that this recipe was tested on a target model and what the outcome was."""
        self.eval_receipts.append(receipt)
        self.updated_at = receipt.dated

    def compound(self, new_fixes: list[FailureFix], stamp: str):
        """Fold a new model's discovered failure->fix pairs back into the recipe.

        This is the compounding mechanic: when a model runs the recipe and finds
        mistakes it still makes, we add the examples that would fix those mistakes
        into the recipe. Future models use an improved recipe with more covering examples.
        """
        # Track which failures are already in the log (avoid duplicates)
        seen = {f.failure_text for f in self.failure_fix_log}
        # Only add new failure->fix pairs we haven't seen before
        added = [f for f in new_fixes if f.failure_text not in seen]
        self.failure_fix_log.extend(added)
        # promote the fixing examples into the influence set (dedup, keep order)
        # Extract the training examples that fix the new failures and add them to the recipe
        for f in added:
            for i in f.fix_indices:
                if i not in self.influence_set:
                    self.influence_set.append(i)
                    self.ordering.append(i)
        # Increment version and timestamp the update
        self.version += 1
        self.updated_at = stamp
        return len(added)

    # -- io -----------------------------------------------------------------
    def to_json(self) -> str:
        """Serialize the recipe to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str):
        """Save the recipe to a JSON file."""
        with open(path, "w") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Recipe":
        """Load a recipe from a JSON file."""
        with open(path) as fh:
            d = json.load(fh)
        # Reconstruct the FailureFix and EvalReceipt objects from JSON dicts
        d["failure_fix_log"] = [FailureFix(**f) for f in d.get("failure_fix_log", [])]
        d["eval_receipts"] = [EvalReceipt(**r) for r in d.get("eval_receipts", [])]
        return cls(**d)
