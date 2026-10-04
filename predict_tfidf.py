#!/usr/bin/env python3
"""Predict likely BNS sections for a case description (TF-IDF model).

    python predict_tfidf.py --text "The accused beat the complainant with a rod and threatened to kill him."
    python predict_tfidf.py --file case.txt
"""
import argparse

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--model", default="model_tfidf/model.joblib")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    text = a.text or open(a.file, encoding="utf-8").read()

    m = joblib.load(a.model)
    p = m["clf"].predict_proba(m["vec"].transform([text]))[0]
    ranked = sorted(zip(m["labels"], p), key=lambda x: -x[1])
    hits = [(s, q) for s, q in ranked if q >= m["threshold"]]
    print("Likely BNS sections:" if hits else "No section passed the threshold; closest:")
    for s, q in (hits or ranked[:a.top]):
        print(f"  BNS {s}   score {q:.2f}")
    print("\n(Decision support only - not legal advice. Scores are not probabilities of guilt.)")


if __name__ == "__main__":
    main()