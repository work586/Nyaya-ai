#!/usr/bin/env python3
"""Predict BNS sections for new case text.
    python predict.py --text "facts of the case ..."
    python predict.py --file case.txt
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train import Model


def predict_sections(text, model_dir="model", top_k=5):
    cfg = json.load(open(Path(model_dir) / "config.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    m = Model(len(cfg["labels"]), cfg["model_name"]).to(dev)
    m.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=dev)); m.eval()
    enc = tok(text, truncation=True, max_length=cfg["max_len"], return_tensors="pt").to(dev)
    with torch.no_grad():
        p = torch.sigmoid(m(enc["input_ids"], enc["attention_mask"]))[0].cpu().tolist()
    ranked = sorted(zip(cfg["labels"], p), key=lambda x: -x[1])
    hits = [(s, round(q, 3)) for s, q in ranked if q >= cfg["threshold"]]
    return hits or [(s, round(q, 3)) for s, q in ranked[:top_k]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text"); ap.add_argument("--file"); ap.add_argument("--model_dir", default="model")
    a = ap.parse_args()
    txt = a.text or open(a.file, encoding="utf-8").read()
    for s, q in predict_sections(txt, a.model_dir):
        print(f"BNS section {s}: {q}")