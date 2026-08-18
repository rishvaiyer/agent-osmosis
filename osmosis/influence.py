"""
Which training examples actually mattered?

The recipe thesis rests on finding the small subset that carries the learning
signal — and on that subset being *similar across different base models* (the
invariance experiment).

What we learned building this: the best *teaching* coreset is neither the
easiest (most confident) examples nor the hardest (near-boundary) ones — training
on either alone plateaus well below the ceiling. What reaches full accuracy with
~15% of the data is a set that *covers the difficulty spectrum*: confident
prototypes that establish the boundary fast, plus harder examples that teach the
subtler features. So selection is a class-balanced, margin-stratified coverage
sample. This is a fast proxy for TracIn / influence-functions / LESS-style
selection; `method` on the recipe records which one produced it.
"""
from __future__ import annotations
import numpy as np
from .models import OsmosisModel


def _probe(model: OsmosisModel, texts, y, passes: int = 3) -> OsmosisModel:
    """Train a fresh model on all training data to score which examples are informative.

    The probe acts like a "trained version" of the base model, so we can measure
    each example's margin (how close to the decision boundary) to see which are tricky.
    """
    y = np.asarray(y)
    # Create a new model with a different seed so it's independent
    probe = OsmosisModel(base_model=model.base_model, seed=7)
    # Train the probe multiple passes over all the data to converge
    for p in range(passes):
        idx = np.random.default_rng(p).permutation(len(texts))
        probe._partial_fit([texts[i] for i in idx], y[idx])
    return probe


def influence_scores(model: OsmosisModel, texts, y, probe_passes: int = 3) -> np.ndarray:
    """Per-example teaching value for the dashboard's recipe inspector: informative
    (near the model's boundary) but correctly labeled scores highest.

    Returns a score 0-1 per example. Higher = more useful for teaching a new model.
    """
    y = np.asarray(y)
    probe = _probe(model, texts, y, probe_passes)
    margin = np.abs(probe.margins(texts))
    # Check which examples the probe got right vs wrong
    correct = (probe.predict(texts) == y)
    # Examples near the boundary (small margin) are informative — they teach the boundary
    info = 1.0 / (1.0 + margin)                 # near-boundary = informative
    # Boost correct examples, penalize mislabeled ones (which are noise or mistakes)
    return _norm(info) * np.where(correct, 1.0, 0.3)


def top_influential(model: OsmosisModel, texts, y, k: int, probe_passes: int = 3,
                    **kw) -> np.ndarray:
    """Class-balanced, margin-stratified coverage coreset of size ~k. Reaches near
    the accuracy ceiling with a fraction of the data — the point of a recipe.

    Strategy: pick k examples that span easy-to-hard per class. Don't just take the
    hardest (they may be noise); don't just take the easiest (they don't teach much).
    Spread across the difficulty spectrum for robust learning.
    """
    y = np.asarray(y)
    probe = _probe(model, texts, y, probe_passes)
    margin = np.abs(probe.margins(texts))
    correct = probe.predict(texts) == y

    # Split k examples evenly between classes (e.g., k/2 positive, k/2 negative)
    classes = np.unique(y)
    per_class = max(2, k // len(classes))
    chosen: list[int] = []
    for c in classes:
        # Find all indices of this class
        ci = np.where(y == c)[0]
        # prefer correctly-labeled; order by margin (easy -> hard) then sample evenly
        # Use only correct examples if available, otherwise use all (even mislabeled)
        keep = ci[correct[ci]] if correct[ci].any() else ci
        # Sort by margin: easy (large margin/far from boundary) first
        order = keep[np.argsort(margin[keep])]
        # Pick evenly spaced examples across the difficulty spectrum
        picks = np.linspace(0, len(order) - 1, min(per_class, len(order))).astype(int)
        chosen.extend(int(i) for i in order[picks])
    return np.array(chosen[:k], dtype=int)


def _norm(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range: map min value to 0, max value to 1."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    # Avoid division by zero with tiny epsilon
    return (x - lo) / (hi - lo + 1e-12)
