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
    """Return `indices` reordered easy->hard by the probe's confidence.

    The idea: train a model on all data, then use its loss scores to rank the given
    indices. Examples with low loss are easy (model is confident); high loss are hard.
    Teaching in easy->hard order helps the learner form better representations.
    """
    # Create a probe model (fresh copy) and train it on all data
    probe = OsmosisModel(base_model=model.base_model, seed=11)
    y = np.asarray(y)
    # Train probe a couple times to converge
    for _ in range(2):
        probe._partial_fit(texts, y)
    # Score just the subset we care about (the influential set)
    sub_texts = [texts[i] for i in indices]
    # Per-example loss: high loss = hard, low loss = easy
    difficulty = probe.per_example_loss(sub_texts, y[indices])  # high loss = hard
    # Sort by difficulty (low->high = easy->hard)
    order = np.argsort(difficulty)  # easy first
    return [int(indices[i]) for i in order]
