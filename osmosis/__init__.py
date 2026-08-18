"""
agent-osmosis — cross-model learning transfer for expedited fine-tuning.

Ship the recipe, not the weights. See docs/eli5.md.
"""
from .data import Task, make_task, make_task_family
from .models import OsmosisModel, BASE_MODELS, Curve
from .recipe import Recipe, FailureFix, EvalReceipt
from .influence import influence_scores, top_influential
from .transfer import extract_recipe, cold_start, warm_start, discover_fixes
from .experiment import warm_vs_cold, compounding_chain, influence_invariance

__all__ = [
    "Task", "make_task", "make_task_family",
    "OsmosisModel", "BASE_MODELS", "Curve",
    "Recipe", "FailureFix", "EvalReceipt",
    "influence_scores", "top_influential",
    "extract_recipe", "cold_start", "warm_start", "discover_fixes",
    "warm_vs_cold", "compounding_chain", "influence_invariance",
]
__version__ = "0.1.0"
