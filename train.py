#!/usr/bin/env python3
"""
Step 2: train a multi-label BNS charge predictor (LegalBERT) on out/dataset.csv.
Settings follow the paper: 8 epochs, batch 4, lr 2e-5, max_len 512, seed 42,
sigmoid + threshold (tuned on validation, default 0.5), pos_weight for rare labels.

    pip install -r requirements.txt
    python train.py --csv out/dataset.csv --out_dir model
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DS(Dataset):
    def __init__(self, texts, Y, tok, max_len):
        self.t, self.Y, self.tok, self.L = texts, Y, tok, max_len

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], truncation=True, max_length=self.L,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.Y[i], dtype=torch.float)}


class Model(torch.nn.Module):
    def __init__(self, n_labels, model_name=MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.drop(h))


@torch.no_grad()
def predict_probs(model, loader, dev):
    model.eval(); P, Y = [], []
    for b in loader:
        lg = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        P.append(torch.sigmoid(lg).cpu().numpy()); Y.append(b["labels"].numpy())
    return np.vstack(P), np.vstack(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/dataset.csv")
    ap.add_argument("--out_dir", default="model")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_label_count", type=int, default=15,
                    help="ignore BNS sections seen in fewer judgments than this")
    ap.add_argument("--min_words", type=int, default=80,
                    help="skip orders with too little text (no facts to learn from)")
    ap.add_argument("--model", default=MODEL_NAME,
                    help="use nlpaueb/legal-bert-small-uncased for a much faster CPU run")
    ap.add_argument("--limit", type=int, default=0,
                    help="quick test: train on only this many orders (e.g. 40)")
    ap.add_argument("--max_words", type=int, default=6000,
                    help="skip very long judgments (usually many cases merged together)")
    args = ap.parse_args()
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    df = pd.read_csv(args.csv).fillna("")
    df = df[(df.sections != "") & (df.n_words >= args.min_words)]
    if "is_combined" in df.columns:  # several petitions decided together -> mixed labels
        df = df[~df.is_combined.astype(str).str.lower().eq("true")]
    df = df[df.n_words <= args.max_words].reset_index(drop=True)
    lists = df.sections.str.split("|")
    counts = pd.Series([s for l in lists for s in l]).value_counts()
    labels = sorted(counts[counts >= args.min_label_count].index, key=lambda x: (int(''.join(c for c in x if c.isdigit())), x))
    if not labels:
        raise SystemExit("No section has enough examples. Collect more judgments or lower --min_label_count.")
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=np.float32)
    for r, l in enumerate(lists):
        for s in l:
            if s in idx: Y[r, idx[s]] = 1
    keep = Y.sum(1) > 0
    df, Y = df[keep].reset_index(drop=True), Y[keep]
    print(f"{len(df)} judgments, {len(labels)} labels")

    tr, tmp = train_test_split(np.arange(len(df)), test_size=0.2, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=args.seed)
    if args.limit:  # quick pipeline test only
        tr, va, te = tr[:args.limit], va[:max(8, args.limit // 4)], te[:max(8, args.limit // 4)]
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = (df.text_model if "text_model" in df.columns else df.text_masked).tolist()
    mk = lambda ids, sh: DataLoader(DS([texts[i] for i in ids], Y[ids], tok, args.max_len),
                                    batch_size=args.batch_size, shuffle=sh)
    L_tr, L_va, L_te = mk(tr, True), mk(va, False), mk(te, False)

    pos = Y[tr].sum(0); pos_weight = torch.tensor((len(tr) - pos) / np.maximum(pos, 1),
                                                  dtype=torch.float).clamp(max=50).to(dev)
    model = Model(len(labels), args.model).to(dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(L_tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best, out = -1, Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0; t0 = time.time()
        for step, b in enumerate(L_tr, 1):
            opt.zero_grad()
            lg = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
            loss = loss_fn(lg, b["labels"].to(dev)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
            if step == 1 or step % 10 == 0 or step == len(L_tr):
                el = time.time() - t0
                eta = el / step * (len(L_tr) - step)
                print(f"  epoch {ep}/{args.epochs}  step {step}/{len(L_tr)}  "
                      f"{el/60:.1f} min elapsed, ~{eta/60:.1f} min left in this epoch", flush=True)
        P, Yv = predict_probs(model, L_va, dev)
        f1 = f1_score(Yv, P >= 0.5, average="micro", zero_division=0)
        print(f"epoch {ep}: loss {tot/len(L_tr):.4f}  val micro-F1 {f1:.4f}")
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    P, Yv = predict_probs(model, L_va, dev)
    thr = max(np.arange(0.1, 0.91, 0.05),
              key=lambda t: f1_score(Yv, P >= t, average="micro", zero_division=0))
    P, Yt = predict_probs(model, L_te, dev)
    pred = P >= thr
    print(f"\nthreshold {thr:.2f}")
    print(f"TEST micro-F1 {f1_score(Yt, pred, average='micro', zero_division=0):.4f}  "
          f"macro-F1 {f1_score(Yt, pred, average='macro', zero_division=0):.4f}")
    json.dump({"labels": labels, "threshold": float(thr), "max_len": args.max_len,
               "model_name": args.model}, open(out / "config.json", "w"), indent=2)
    print("saved to", out)


if __name__ == "__main__":
    main()