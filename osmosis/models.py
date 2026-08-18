"""
Models for agent-osmosis.

An `OsmosisModel` = a frozen *encoder* (the base model's internal representation)
+ a trainable *head* (what fine-tuning actually moves). Different encoders stand
in for different base-model architectures: they "see" the same text through
different features, exactly the reason raw weights don't transfer between models
but a *recipe* (which examples, in which order) can.

The head is an SGD classifier trained with `partial_fit`, so we can measure
learning one mini-batch at a time and get real steps-to-accuracy curves. Swap this
whole file for `transformers` + `peft` and the rest of the engine is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


# Registry of "base models". Each is an encoder config -> a different representation.
BASE_MODELS = {
    "osmo-base-a": dict(n_features=2**14, ngram_range=(1, 1), alternate_sign=False, seed=1),
    "osmo-base-b": dict(n_features=2**13, ngram_range=(1, 2), alternate_sign=False, seed=2),
    "osmo-base-c": dict(n_features=2**12, ngram_range=(1, 2), alternate_sign=True,  seed=3),
    "osmo-wide-d": dict(n_features=2**15, ngram_range=(1, 3), alternate_sign=False, seed=4),
}


@dataclass
class Curve:
    """A learning curve from streaming training."""
    seen: list[int] = field(default_factory=list)      # cumulative training examples
    val_acc: list[float] = field(default_factory=list)  # validation accuracy at each point

    def steps_to_target(self, target: float) -> int | None:
        for s, a in zip(self.seen, self.val_acc):
            if a >= target:
                return s
        return None


class OsmosisModel:
    """A model with a frozen encoder (how text is converted to numbers) and a trainable
    classification head (like a LoRA adapter, but simpler). This lets us see learning happen
    step by step as we train on examples one batch at a time."""
    def __init__(self, base_model: str = "osmo-base-a", lr: float = 0.15, seed: int = 0):
        if base_model not in BASE_MODELS:
            raise KeyError(f"unknown base model {base_model!r}; choices: {list(BASE_MODELS)}")
        cfg = BASE_MODELS[base_model]
        self.base_model = base_model
        # Encoder: converts text to numbers (features) in a fixed way. This part doesn't change.
        self.encoder = HashingVectorizer(
            n_features=cfg["n_features"], ngram_range=cfg["ngram_range"],
            alternate_sign=cfg["alternate_sign"], norm="l2",
        )
        # Trainable head: learns to classify the encoded text. This is what fine-tuning trains.
        # the trainable head — this is the analog of the LoRA adapter
        self.head = SGDClassifier(
            loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=lr,
            random_state=cfg["seed"] + seed,
        )
        self._init_seed = cfg["seed"] + seed
        self._fitted = False

    # -- encoding -----------------------------------------------------------
    def encode(self, texts: list[str]):
        """Convert a list of text strings to their numeric representation (features)."""
        return self.encoder.transform(texts)

    # -- training -----------------------------------------------------------
    def _partial_fit(self, texts, y):
        """Train the head on one batch of texts. On first call, initialize with class labels."""
        X = self.encode(texts)
        if not self._fitted:
            # First time: tell the classifier about both classes (0 and 1)
            self.head.partial_fit(X, y, classes=np.array([0, 1]))
            self._fitted = True
        else:
            # After first time, just update with new data
            self.head.partial_fit(X, y)

    def train_stream(self, texts, y, val_texts, val_y, batch: int = 16,
                     eval_every: int = 32, curve: Curve | None = None,
                     start_seen: int = 0) -> Curve:
        """Stream examples in mini-batches; log val accuracy as we go.

        This mimics real online learning: feed examples in order, measure accuracy at
        checkpoints, and track how many examples it took to reach certain accuracies.
        """
        curve = curve or Curve()
        y = np.asarray(y)
        seen = start_seen
        # Process examples in batches
        for i in range(0, len(texts), batch):
            bt = texts[i:i + batch]
            by = y[i:i + batch]
            if len(bt) == 0:
                continue
            # Train on this batch
            self._partial_fit(bt, by)
            seen += len(bt)
            # Check if we've hit an eval checkpoint (every N examples)
            if seen // eval_every != (seen - len(bt)) // eval_every:
                curve.seen.append(seen)
                curve.val_acc.append(self.accuracy(val_texts, val_y))
        # always record the final point after all examples are seen
        curve.seen.append(seen)
        curve.val_acc.append(self.accuracy(val_texts, val_y))
        return curve

    def warm_prefit(self, texts, y, passes: int = 2):
        """Pre-train on a curated recipe subset before the main stream. This is
        the 'warm start': the head arrives already shaped by the transferred recipe.

        We train multiple passes over the recipe data to let the head absorb the signal
        before learning from the full dataset.
        """
        y = np.asarray(y)
        # Train multiple times over the recipe coreset to let it sink in
        for _ in range(passes):
            self._partial_fit(texts, y)

    # -- inference ----------------------------------------------------------
    def predict(self, texts):
        """Predict the class label (0 or 1) for each text."""
        return self.head.predict(self.encode(texts))

    def margins(self, texts):
        """Distance to the decision boundary. Small |margin| = near-boundary.

        This tells us how confident the model is. Close to 0 means uncertain/near the edge.
        """
        return self.head.decision_function(self.encode(texts))

    def accuracy(self, texts, y) -> float:
        """Compute the fraction of correct predictions."""
        return float(np.mean(self.predict(texts) == np.asarray(y)))

    def per_example_loss(self, texts, y):
        """Logistic loss per example — the signal influence/curriculum build on.

        High loss = example is confusing to the model (near the boundary or wrong).
        This metric is used to decide which examples are worth learning from.
        """
        y = np.asarray(y)
        m = self.margins(texts)
        # Convert margins to probabilities via sigmoid function
        p = 1.0 / (1.0 + np.exp(-m))
        # Clip to avoid log(0) numerical errors
        p = np.clip(p, 1e-6, 1 - 1e-6)
        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
