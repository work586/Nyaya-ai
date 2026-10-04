#!/usr/bin/env python3
"""
Step 1b: look at the dataset BEFORE training.

    python inspect_data.py

Prints:
  - how many judgments are usable at different --min_label_count values
  - the duplicates that were removed with the lowest similarity (check these by hand)
  - a few random orders so you can see whether they contain the facts of the case
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--min_words", type=int, default=80)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out_dir)

    df = pd.read_csv(out / "dataset.csv").fillna("")
    print(f"Unique judgments            : {len(df)}")
    has = df[df.sections != ""]
    print(f"With at least one BNS section: {len(has)}")
    usable = has[has.n_words >= a.min_words]
    print(f"...and >= {a.min_words} words        : {len(usable)}")
    print(f"Average sections per order  : {usable.n_sections.mean():.2f}\n")

    lists = usable.sections.str.split("|")
    counts = pd.Series([s for l in lists for s in l]).value_counts()
    print("If you train with --min_label_count N:")
    print(f"{'N':>5} {'labels kept':>12} {'orders kept':>12}")
    for n in (5, 10, 15, 30, 50):
        keep = set(counts[counts >= n].index)
        docs = sum(any(s in keep for s in l) for l in lists)
        print(f"{n:>5} {len(keep):>12} {docs:>12}")
    print()

    dpath = out / "duplicates_report.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        print("Duplicates removed by reason:")
        print(d.reason.value_counts().to_string(), "\n")
        near = d[d.reason == "near_duplicate"].sort_values("similarity").head(8)
        if len(near):
            print("Least-similar 'near duplicates' - open both PDFs and check they are really the same order:")
            for _, r in near.iterrows():
                print(f"  {r.similarity:.2f}  dropped: {r.dropped}   kept: {r.kept_copy}")
            print("  (if these are different orders, re-run build_dataset.py with --dup_threshold 0.95)\n")

    random.seed(1)
    print("=" * 70)
    print("Random orders (masked text, first 700 characters). Do they describe what happened?")
    for i in random.sample(range(len(usable)), min(a.samples, len(usable))):
        r = usable.iloc[i]
        print("-" * 70)
        print(f"{r.source_file} | sections: {r.sections} | words: {r.n_words}")
        print(r.text_masked[:700])


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Step 1b: look at the dataset BEFORE training.

    python inspect_data.py

Prints:
  - how many judgments are usable at different --min_label_count values
  - the duplicates that were removed with the lowest similarity (check these by hand)
  - a few random orders so you can see whether they contain the facts of the case
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--min_words", type=int, default=80)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out_dir)

    df = pd.read_csv(out / "dataset.csv").fillna("")
    print(f"Unique judgments            : {len(df)}")
    has = df[df.sections != ""]
    print(f"With at least one BNS section: {len(has)}")
    usable = has[has.n_words >= a.min_words]
    print(f"...and >= {a.min_words} words        : {len(usable)}")
    print(f"Average sections per order  : {usable.n_sections.mean():.2f}\n")

    lists = usable.sections.str.split("|")
    counts = pd.Series([s for l in lists for s in l]).value_counts()
    print("If you train with --min_label_count N:")
    print(f"{'N':>5} {'labels kept':>12} {'orders kept':>12}")
    for n in (5, 10, 15, 30, 50):
        keep = set(counts[counts >= n].index)
        docs = sum(any(s in keep for s in l) for l in lists)
        print(f"{n:>5} {len(keep):>12} {docs:>12}")
    print()

    dpath = out / "duplicates_report.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        print("Duplicates removed by reason:")
        print(d.reason.value_counts().to_string(), "\n")
        near = d[d.reason == "near_duplicate"].sort_values("similarity").head(8)
        if len(near):
            print("Least-similar 'near duplicates' - open both PDFs and check they are really the same order:")
            for _, r in near.iterrows():
                print(f"  {r.similarity:.2f}  dropped: {r.dropped}   kept: {r.kept_copy}")
            print("  (if these are different orders, re-run build_dataset.py with --dup_threshold 0.95)\n")

    random.seed(1)
    print("=" * 70)
    print("Random orders (masked text, first 700 characters). Do they describe what happened?")
    for i in random.sample(range(len(usable)), min(a.samples, len(usable))):
        r = usable.iloc[i]
        print("-" * 70)
        print(f"{r.source_file} | sections: {r.sections} | words: {r.n_words}")
        print(r.text_masked[:700])


if __name__ == "__main__":
    main()