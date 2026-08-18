"""
What order should the examples be taught in?

Easy-to-hard (a classic curriculum) converges faster and more stably than random
order on the kind of near-boundary task we build. The recipe stores this ordering
so a new model replays the *sequence*, not just the set.
"""
from __future__ import annotations
import numpy as np
from .models import OsmosisModel


def order_easy_to_hard(model: OsmosisModel, texts, y, indices) -> list[int]:
    """Return `indices` reordered easy->hard by the probe's confidence."""
    probe = OsmosisModel(base_model=model.base_model, seed=11)
    y = np.asarray(y)
    for _ in range(2):
        probe._partial_fit(texts, y)
    sub_texts = [texts[i] for i in indices]
    difficulty = probe.per_example_loss(sub_texts, y[indices])  # high loss = hard
    order = np.argsort(difficulty)  # easy first
    return [int(indices[i]) for i in order]
