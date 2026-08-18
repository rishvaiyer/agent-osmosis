"""
Tasks for agent-osmosis.

A Task is a small, human-readable text-classification problem. We generate them
from templates so they need zero downloads, run in milliseconds, and stay legible
in the dashboard (you can *read* the examples in a recipe's failure->fix log).

The point of the MVP is the *transfer machinery*, not the task. Every task here
implements the same interface, and a real LLM fine-tuning task can be dropped in
behind the same `Task` shape without touching the rest of the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Task:
    name: str
    texts: list[str]
    labels: np.ndarray            # int array, shape (n,)
    classes: list[str]            # human names per label index
    train_idx: np.ndarray
    val_idx: np.ndarray
    difficulty: np.ndarray        # per-example latent difficulty in [0,1]
    meta: dict = field(default_factory=dict)

    def slice(self, idx):
        return [self.texts[i] for i in idx], self.labels[idx]

    @property
    def n(self) -> int:
        return len(self.texts)


# --- template pieces -------------------------------------------------------

_SUBJECTS = ["the update", "your API", "the billing page", "the mobile app",
             "our dashboard", "the export", "the login flow", "the webhook"]
_POS = ["works perfectly", "is fast now", "saved me hours", "is exactly right",
        "finally makes sense", "is a huge improvement", "runs smoothly"]
_NEG = ["keeps crashing", "lost my data", "is broken again", "times out",
        "makes no sense", "double charged me", "won't load at all"]
_HEDGE = ["honestly", "i guess", "sort of", "kind of", "more or less", "somewhat"]
_MILD_POS = ["is fine", "is okay", "is decent", "does the job", "is alright"]
_MILD_NEG = ["is meh", "is clunky", "is sluggish", "is annoying", "is fiddly"]


def _templated_sentiment(rng, hard: bool):
    """Positive vs negative feedback. Easy examples use strong, unambiguous words;
    `hard` examples are hedged and use mild same-polarity words — genuinely harder
    (they need more data to pin down) but still learnable, so the ceiling is high."""
    # Randomly pick label: 1 for positive, 0 for negative
    label = int(rng.random() < 0.5)  # 1 = positive, 0 = negative
    subj = rng.choice(_SUBJECTS)
    if not hard:
        # Easy case: use strong, unambiguous language (easy for a learner to classify)
        core = f"{subj} {rng.choice(_POS if label == 1 else _NEG)}"
        return core, label, 0.15
    # hard: hedged + mild wording, label-consistent (no surface flip)
    # Harder case: use hedging words + milder sentiment to make it trickier
    mild = rng.choice(_MILD_POS if label == 1 else _MILD_NEG)
    core = f"{rng.choice(_HEDGE)} {subj} {mild}"
    return core, label, 0.7


def make_task(name: str = "support-sentiment", n: int = 600, seed: int = 0,
              hard_fraction: float = 0.35, val_fraction: float = 0.25) -> Task:
    """Build a templated text task. `hard_fraction` controls how many examples
    are near-boundary (the ones that carry the learning signal)."""
    # Set up random number generator with seed for reproducibility
    rng = np.random.default_rng(seed)
    texts, labels, diff = [], [], []
    # Generate n examples, mixing easy and hard difficulty levels
    for _ in range(n):
        # Decide if this example should be hard or easy based on hard_fraction
        hard = rng.random() < hard_fraction
        t, y, d = _templated_sentiment(rng, hard)
        texts.append(t)
        labels.append(y)
        # Add small noise to difficulty scores for realism
        diff.append(d + float(rng.normal(0, 0.03)))
    labels = np.array(labels, dtype=int)
    # Clamp all difficulty scores to [0,1] range
    diff = np.clip(np.array(diff), 0, 1)

    # Split data into training and validation sets randomly
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return Task(
        name=name, texts=texts, labels=labels,
        classes=["negative", "positive"],
        train_idx=train_idx, val_idx=val_idx, difficulty=diff,
        meta={"seed": seed, "hard_fraction": hard_fraction},
    )


def make_task_family(n_tasks: int = 3, n: int = 600, base_seed: int = 0) -> list[Task]:
    """A family of related tasks (successive 'generations' / model releases share
    the task, different seeds). Used by the compounding experiment."""
    return [make_task(name=f"gen-{i+1}", n=n, seed=base_seed + i) for i in range(n_tasks)]
