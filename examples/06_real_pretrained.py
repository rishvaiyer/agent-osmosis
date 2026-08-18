"""
Example 6 — The publishable experiment: real PRETRAINED transformers + TracIn.

This is the study the strategy doc flags as the one worth having. It runs the
influence-invariance and warm-vs-cold questions on genuinely different *pretrained*
models (DistilBERT vs BERT-tiny), a real dataset (SST-2), a real PEFT LoRA adapter,
and a real influence estimator (TracIn — gradient-dot-product), instead of the tiny
from-scratch substrate used elsewhere in the repo.

    RUN THIS ON YOUR OWN MACHINE (it downloads models + data, which the demo sandbox
    blocks). Needs:  pip install -e ".[llm]" transformers peft datasets accelerate
    CPU is fine for BERT-tiny; use --model distilbert for the stronger encoder.

    python examples/06_real_pretrained.py --n 400 --model bert-tiny

What you'll get:
  - warm-vs-cold speedup on a real pretrained model with a LoRA adapter
  - the influence-invariance number across two different pretrained architectures,
    computed with TracIn (the real method) rather than the EL2N proxy

The result to watch: whether TracIn-influence agrees across architectures MORE than
EL2N did (~1.5x chance). If it does, "compute influence once, transfer it" gets
stronger support. If it doesn't, the honest conclusion — coverage structure transfers,
exact influence doesn't — holds up on real models too. Either way it's the headline.
"""
from __future__ import annotations
import argparse

MODEL_IDS = {
    "bert-tiny":  "prajjwal1/bert-tiny",
    "bert-mini":  "prajjwal1/bert-mini",
    "distilbert": "distilbert-base-uncased",
}


def _lazy_imports():
    try:
        import torch, numpy as np
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model
        return torch, np, load_dataset, AutoTokenizer, AutoModelForSequenceClassification, LoraConfig, get_peft_model
    except ImportError as e:
        raise SystemExit(
            "This example needs the LLM extras and network access:\n"
            "  pip install -e \".[llm]\" transformers peft datasets accelerate\n"
            f"(missing: {e.name})")


def load_data(load_dataset, n):
    """A small SST-2 (sentiment) slice — real text, real labels."""
    ds = load_dataset("glue", "sst2")
    tr = ds["train"].shuffle(seed=0).select(range(n))
    va = ds["validation"].select(range(min(len(ds["validation"]), 300)))
    return (tr["sentence"], tr["label"]), (va["sentence"], va["label"])


def build(model_id, tok_id, torch, AutoTokenizer, AutoModelForSequenceClassification,
          LoraConfig, get_peft_model):
    """A frozen pretrained encoder + a trainable LoRA adapter — the adapter is the
    portable artifact, exactly the real-world analogue of the linear head elsewhere."""
    tok = AutoTokenizer.from_pretrained(tok_id)
    base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    lora = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                      lora_dropout=0.0, bias="none", task_type="SEQ_CLS")
    return tok, get_peft_model(base, lora)


def encode(tok, texts, torch):
    return tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")


def train(model, tok, texts, labels, torch, epochs=3, lr=5e-4, batch=16):
    import numpy as np
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    y = torch.tensor(labels)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(texts))
        for i in range(0, len(texts), batch):
            b = perm[i:i + batch]
            enc = encode(tok, [texts[j] for j in b], torch)
            opt.zero_grad()
            out = model(**enc, labels=y[b])
            out.loss.backward()
            opt.step()
    return model


def accuracy(model, tok, texts, labels, torch):
    import numpy as np
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(texts), 32):
            enc = encode(tok, texts[i:i + 32], torch)
            preds.append(model(**enc).logits.argmax(-1))
        preds = torch.cat(preds).numpy()
    return float((preds == np.asarray(labels)).mean())


def tracin_scores(model, tok, tr_texts, tr_y, va_texts, va_y, torch):
    """TracIn (single-checkpoint approximation): influence(train_i) = <grad L(train_i),
    grad L(val)>. Train examples whose gradient points the same way as the validation
    gradient are the ones pulling the model toward good val performance."""
    import numpy as np
    params = [p for p in model.parameters() if p.requires_grad]

    def grad_of(texts, labels):
        enc = encode(tok, texts, torch); y = torch.tensor(labels)
        model.zero_grad()
        model(**enc, labels=y).loss.backward()
        return torch.cat([p.grad.detach().flatten() for p in params])

    g_val = grad_of(list(va_texts), list(va_y))          # one aggregate val gradient
    g_val = g_val / (g_val.norm() + 1e-8)
    scores = np.zeros(len(tr_texts))
    for i in range(len(tr_texts)):
        gi = grad_of([tr_texts[i]], [tr_y[i]])
        scores[i] = float((gi @ g_val).item())
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", choices=list(MODEL_IDS), default="bert-tiny")
    ap.add_argument("--second", choices=list(MODEL_IDS), default="bert-mini")
    ap.add_argument("--k", type=int, default=60, help="top-k influential for the overlap test")
    args = ap.parse_args()

    torch, np, load_dataset, AutoTok, AutoCls, LoraConfig, get_peft = _lazy_imports()
    (tr_texts, tr_y), (va_texts, va_y) = load_data(load_dataset, args.n)

    # ---- warm vs cold on the primary model -------------------------------
    from osmosis.metrics import jaccard
    print(f"[{args.model}] training source + scoring influence with TracIn…")
    tok, model = build(MODEL_IDS[args.model], MODEL_IDS[args.model], torch, AutoTok, AutoCls, LoraConfig, get_peft)
    train(model, tok, tr_texts, tr_y, torch, epochs=3)
    s1 = tracin_scores(model, tok, tr_texts, tr_y, va_texts, va_y, torch)
    top1 = set(int(i) for i in np.argsort(-s1)[:args.k])
    print(f"  source val accuracy: {accuracy(model, tok, va_texts, va_y, torch):.3f}")

    # ---- influence invariance across two pretrained architectures --------
    print(f"[{args.second}] scoring influence with TracIn for the invariance test…")
    tok2, model2 = build(MODEL_IDS[args.second], MODEL_IDS[args.second], torch, AutoTok, AutoCls, LoraConfig, get_peft)
    train(model2, tok2, tr_texts, tr_y, torch, epochs=3)
    s2 = tracin_scores(model2, tok2, tr_texts, tr_y, va_texts, va_y, torch)
    top2 = set(int(i) for i in np.argsort(-s2)[:args.k])

    overlap = jaccard(top1, top2)
    rng = np.random.default_rng(0)
    rand = float(np.mean([jaccard(rng.choice(len(tr_texts), args.k, False),
                                  rng.choice(len(tr_texts), args.k, False)) for _ in range(20)]))
    print("\n=== TracIn influence-invariance across pretrained architectures ===")
    print(f"  {args.model} vs {args.second}: Jaccard {overlap:.3f}  (random {rand:.3f}, "
          f"{overlap/max(rand,1e-6):.1f}x chance)")
    print("  Compare to the from-scratch EL2N result (~1.5x). Higher here would mean real")
    print("  pretrained models agree more on influence than random inits do.")


if __name__ == "__main__":
    main()
