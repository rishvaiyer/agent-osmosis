"""
Real-transformer backend (no downloads).

The default engine uses hashing-encoder linear models so the whole demo runs in
seconds. This module swaps in *actual transformers* — real multi-head self-attention,
real backprop — built from scratch in PyTorch so nothing needs to be downloaded
(model hubs are unreachable from the sandbox anyway). It answers the question the
strategy doc flags as the one worth proving: do the recipe mechanics — and especially
influence invariance — still hold across genuinely different transformer architectures?

Different `TORCH_BASES` entries are different transformer configs (depth / heads /
width / seed). They are heterogeneous models in the sense that matters here: distinct
parameter spaces and distinct learned representations.

    from osmosis.torch_backend import real_warm_vs_cold, real_influence_invariance
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .data import Task
from .metrics import jaccard, mean_ci

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

TORCH_BASES = {
    "tf-small-a":  dict(d=32, heads=2, layers=1, seed=1),
    "tf-small-b":  dict(d=24, heads=3, layers=1, seed=2),
    "tf-deep-c":   dict(d=32, heads=4, layers=2, seed=3),
    "tf-wide-d":   dict(d=48, heads=2, layers=1, seed=4),
}
SEQ_LEN = 14


# ------------------------------------------------------------ tokenizer
def build_vocab(texts) -> dict:
    vocab = {"<pad>": 0}
    for t in texts:
        for w in t.split():
            vocab.setdefault(w, len(vocab))
    return vocab


def encode(texts, vocab, L=SEQ_LEN) -> torch.Tensor:
    X = torch.zeros(len(texts), L, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, w in enumerate(t.split()[:L]):
            X[i, j] = vocab.get(w, 0)
    return X


# ------------------------------------------------------------ model
class _Net(nn.Module):
    def __init__(self, V, d, heads, layers, L=SEQ_LEN, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        mask = (x == 0)
        z = self.emb(x) + self.pos[:, : x.size(1)]
        z = self.tf(z, src_key_padding_mask=mask)
        z = z.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
        return self.head(z)


class TorchModel:
    """Same surface the engine expects: train_stream / warm_prefit / predict /
    margins / per_example_loss — backed by a real transformer."""

    def __init__(self, base: str, vocab: dict, lr: float = 5e-3, seed: int = 0):
        cfg = TORCH_BASES[base]
        self.base_model = base
        self.vocab = vocab
        self.net = _Net(len(vocab), cfg["d"], cfg["heads"], cfg["layers"], seed=cfg["seed"] + seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lossf = nn.CrossEntropyLoss()

    def _step(self, X, y):
        self.net.train()
        self.opt.zero_grad()
        loss = self.lossf(self.net(X), y)
        loss.backward()
        self.opt.step()

    def train_stream(self, texts, y, val_texts, val_y, batch=16, eval_every=32,
                     curve=None, start_seen=0):
        from .models import Curve
        curve = curve or Curve()
        X = encode(texts, self.vocab); y = torch.tensor(np.asarray(y))
        seen = start_seen
        for i in range(0, len(X), batch):
            self._step(X[i:i + batch], y[i:i + batch])
            seen += len(X[i:i + batch])
            if seen // eval_every != (seen - len(X[i:i + batch])) // eval_every:
                curve.seen.append(seen); curve.val_acc.append(self.accuracy(val_texts, val_y))
        curve.seen.append(seen); curve.val_acc.append(self.accuracy(val_texts, val_y))
        return curve

    def warm_prefit(self, texts, y, passes=4):
        X = encode(texts, self.vocab); y = torch.tensor(np.asarray(y))
        for _ in range(passes):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 16):
                b = perm[i:i + 16]; self._step(X[b], y[b])

    def fit_epochs(self, texts, y, epochs=6):
        self.warm_prefit(texts, y, passes=epochs)

    @torch.no_grad()
    def _logits(self, texts):
        self.net.eval()
        return self.net(encode(texts, self.vocab))

    def predict(self, texts):
        return self._logits(texts).argmax(1).numpy()

    def margins(self, texts):
        lg = self._logits(texts)
        return (lg[:, 1] - lg[:, 0]).numpy()

    def accuracy(self, texts, y):
        return float((self.predict(texts) == np.asarray(y)).mean())

    def per_example_loss(self, texts, y):
        lg = self._logits(texts)
        return nn.functional.cross_entropy(lg, torch.tensor(np.asarray(y)), reduction="none").numpy()


# ------------------------------------------------------------ influence (EL2N-style)
def el2n_scores(model: TorchModel, texts, y) -> np.ndarray:
    """EL2N: L2 norm of the error vector (softmax - onehot). One forward pass, no
    gradients — the standard cheap data-importance signal."""
    with torch.no_grad():
        p = torch.softmax(model._logits(texts), dim=1).numpy()
    onehot = np.eye(2)[np.asarray(y)]
    return np.linalg.norm(p - onehot, axis=1)


def coverage_coreset(model: TorchModel, texts, y, k: int) -> np.ndarray:
    """Class-balanced, margin-stratified coverage coreset (same principle as the
    linear engine): span the difficulty spectrum, don't just take the hardest."""
    y = np.asarray(y); margin = np.abs(model.margins(texts))
    per_class = max(2, k // 2)
    chosen = []
    for c in [0, 1]:
        ci = np.where(y == c)[0]
        order = ci[np.argsort(margin[ci])]
        picks = np.linspace(0, len(order) - 1, min(per_class, len(order))).astype(int)
        chosen += [int(i) for i in order[picks]]
    return np.array(chosen[:k])


# ------------------------------------------------------------ experiments
def _prep(task: Task):
    tr, trY = task.slice(task.train_idx)
    va, vaY = task.slice(task.val_idx)
    vocab = build_vocab(task.texts)
    return tr, np.asarray(trY), va, np.asarray(vaY), vocab


def real_warm_vs_cold(task: Task, source="tf-small-a", target="tf-small-b",
                      budget_frac=0.2, target_acc=0.85, trials=4, lr=5e-3):
    tr, trY, va, vaY, vocab = _prep(task)
    k = max(6, int(len(tr) * budget_frac))

    # source learns, then we pick the coverage coreset in curriculum (easy->hard) order
    src = TorchModel(source, vocab, lr=lr); src.fit_epochs(tr, trY, epochs=6)
    core = coverage_coreset(src, tr, trY, k)
    core = core[np.argsort(np.abs(src.margins([tr[i] for i in core])))]  # easy->hard

    warm_steps, cold_steps, warm_curves, cold_curves = [], [], [], []
    for s in range(trials):
        # cold
        cm = TorchModel(target, vocab, lr=lr, seed=s)
        rng = np.random.default_rng(s); perm = rng.permutation(len(tr))
        cc = cm.train_stream([tr[i] for i in perm], trY[perm], va, vaY)
        # warm
        wm = TorchModel(target, vocab, lr=lr, seed=s)
        wm.warm_prefit([tr[i] for i in core], trY[core], passes=4)
        from .models import Curve
        wc = Curve(seen=[len(core)], val_acc=[wm.accuracy(va, vaY)])
        rest = [i for i in rng.permutation(len(tr)) if i not in set(core.tolist())]
        wc = wm.train_stream([tr[i] for i in rest], trY[rest], va, vaY, curve=wc, start_seen=len(core))
        cold_steps.append(cc.steps_to_target(target_acc)); warm_steps.append(wc.steps_to_target(target_acc))
        cold_curves.append(cc); warm_curves.append(wc)

    cm_, ch = mean_ci(cold_steps); wm_, wh = mean_ci(warm_steps)
    return {
        "cold_mean": cm_, "cold_ci": ch, "warm_mean": wm_, "warm_ci": wh,
        "speedup": (cm_ / wm_) if (cm_ and wm_) else None,
        "coreset": len(core), "target_acc": target_acc,
        "warm_steps": warm_steps, "cold_steps": cold_steps,
    }


def real_influence_invariance(task: Task, bases=None, budget_frac=0.2, lr=5e-3):
    """The result worth having: do *different real transformers* agree on which
    examples are most influential (highest EL2N)?"""
    tr, trY, va, vaY, vocab = _prep(task)
    bases = list(bases or TORCH_BASES.keys())
    k = max(6, int(len(tr) * budget_frac))
    tops = {}
    for b in bases:
        m = TorchModel(b, vocab, lr=lr); m.fit_epochs(tr, trY, epochs=5)
        s = el2n_scores(m, tr, trY)
        tops[b] = set(int(i) for i in np.argsort(-s)[:k])
    matrix = [[round(jaccard(tops[a], tops[c]), 3) for c in bases] for a in bases]
    n = len(tr); rng = np.random.default_rng(0)
    rand = float(np.mean([jaccard(rng.choice(n, k, False), rng.choice(n, k, False)) for _ in range(20)]))
    off = [matrix[i][j] for i in range(len(bases)) for j in range(len(bases)) if i != j]
    return {"bases": bases, "k": k, "matrix": matrix,
            "mean_overlap": round(float(np.mean(off)), 3), "random_overlap": round(rand, 3),
            "verdict": "invariant" if np.mean(off) > 3 * rand else "weakly-invariant"}
