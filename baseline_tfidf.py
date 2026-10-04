#!/usr/bin/env python3
"""
Baseline: TF-IDF + logistic regression on the SAME orders and SAME split as train.py.
Runs in about a minute on a CPU. Use it to judge whether LegalBERT is really helping,
and to see which BNS sections can be predicted at all.

    python baseline_tfidf.py --min_label_count 30
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/dataset.csv")
    ap.add_argument("--min_label_count", type=int, default=30)
    ap.add_argument("--min_words", type=int, default=80)
    ap.add_argument("--max_words", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--save", default="", help="folder to save the final model, e.g. model_tfidf")
    args = ap.parse_args()

    # ---- identical data preparation to train.py ----
    df = pd.read_csv(args.csv).fillna("")
    df = df[(df.sections != "") & (df.n_words >= args.min_words)]
    if "is_combined" in df.columns:
        df = df[~df.is_combined.astype(str).str.lower().eq("true")]
    df = df[df.n_words <= args.max_words].reset_index(drop=True)
    lists = df.sections.str.split("|")
    counts = pd.Series([s for l in lists for s in l]).value_counts()
    labels = sorted(counts[counts >= args.min_label_count].index,
                    key=lambda x: (int(''.join(c for c in x if c.isdigit())), x))
    idx = {l: i for i, l in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=int)
    for r, l in enumerate(lists):
        for s in l:
            if s in idx:
                Y[r, idx[s]] = 1
    keep = Y.sum(1) > 0
    df, Y = df[keep].reset_index(drop=True), Y[keep]
    texts = (df.text_model if "text_model" in df.columns else df.text_masked).tolist()
    print(f"{len(df)} judgments, {len(labels)} labels")

    tr, tmp = train_test_split(np.arange(len(df)), test_size=0.2, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.5, random_state=args.seed)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                          sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva, Xte = vec.transform([texts[i] for i in va]), vec.transform([texts[i] for i in te])

    clf = OneVsRestClassifier(
        LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000), n_jobs=-1)
    clf.fit(Xtr, Y[tr])
    Pva, Pte = clf.predict_proba(Xva), clf.predict_proba(Xte)

    thr = max(np.arange(0.1, 0.91, 0.05),
              key=lambda t: f1_score(Y[va], Pva >= t, average="micro", zero_division=0))
    pred = Pte >= thr
    print(f"\nthreshold {thr:.2f}  (picked on the validation set)")
    print(f"TF-IDF TEST micro-F1 {f1_score(Y[te], pred, average='micro', zero_division=0):.4f}  "
          f"macro-F1 {f1_score(Y[te], pred, average='macro', zero_division=0):.4f}")

    allpos = np.ones_like(Y[te])
    print(f"'Predict every section for every order' scores micro-F1 "
          f"{f1_score(Y[te], allpos, average='micro', zero_division=0):.4f}  <- the floor to beat")

    print(f"\nPer-section results on the test set ({len(te)} orders):")
    f1s = f1_score(Y[te], pred, average=None, zero_division=0)
    rows = pd.DataFrame({"bns_section": labels, "orders_in_test": Y[te].sum(0),
                         "orders_in_all_data": [counts[l] for l in labels], "F1": f1s.round(2)})
    print(rows.sort_values("F1", ascending=False).to_string(index=False))

    if args.save:  # final model: refit on ALL orders (more data helps), keep tuned threshold
        import joblib
        from pathlib import Path
        vec_all = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                  sublinear_tf=True, lowercase=True)
        X_all = vec_all.fit_transform(texts)
        clf_all = OneVsRestClassifier(
            LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000), n_jobs=-1)
        clf_all.fit(X_all, Y)
        Path(args.save).mkdir(exist_ok=True, parents=True)
        joblib.dump({"vec": vec_all, "clf": clf_all, "labels": labels, "threshold": float(thr)},
                    Path(args.save) / "model.joblib")
        print(f"\nSaved final model to {args.save}/model.joblib  (trained on all {len(df)} orders)")


if __name__ == "__main__":
    main()