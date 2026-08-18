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
    y = np.asarray(y)
    probe = OsmosisModel(base_model=model.base_model, seed=7)
    for p in range(passes):
        idx = np.random.default_rng(p).permutation(len(texts))
        probe._partial_fit([texts[i] for i in idx], y[idx])
    return probe


def influence_scores(model: OsmosisModel, texts, y, probe_passes: int = 3) -> np.ndarray:
    """Per-example teaching value for the dashboard's recipe inspector: informative
    (near the model's boundary) but correctly labeled scores highest."""
    y = np.asarray(y)
    probe = _probe(model, texts, y, probe_passes)
    margin = np.abs(probe.margins(texts))
    correct = (probe.predict(texts) == y)
    info = 1.0 / (1.0 + margin)                 # near-boundary = informative
    return _norm(info) * np.where(correct, 1.0, 0.3)


def top_influential(model: OsmosisModel, texts, y, k: int, probe_passes: int = 3,
                    **kw) -> np.ndarray:
    """Class-balanced, margin-stratified coverage coreset of size ~k. Reaches near
    the accuracy ceiling with a fraction of the data — the point of a recipe."""
    y = np.asarray(y)
    probe = _probe(model, texts, y, probe_passes)
    margin = np.abs(probe.margins(texts))
    correct = probe.predict(texts) == y

    classes = np.unique(y)
    per_class = max(2, k // len(classes))
    chosen: list[int] = []
    for c in classes:
        ci = np.where(y == c)[0]
        # prefer correctly-labeled; order by margin (easy -> hard) then sample evenly
        keep = ci[correct[ci]] if correct[ci].any() else ci
        order = keep[np.argsort(margin[keep])]
        picks = np.linspace(0, len(order) - 1, min(per_class, len(order))).astype(int)
        chosen.extend(int(i) for i in order[picks])
    return np.array(chosen[:k], dtype=int)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)
