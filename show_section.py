#!/usr/bin/env python3
"""
Show where a BNS section appears in your orders, to check that a label is real.

    python show_section.py 269
    python show_section.py 269 --n 8
"""
import argparse
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section")
    ap.add_argument("--csv", default="out/dataset.csv")
    ap.add_argument("--n", type=int, default=5, help="how many orders to show")
    a = ap.parse_args()

    df = pd.read_csv(a.csv).fillna("")
    has = df[df.sections.apply(lambda s: a.section in s.split("|"))]
    print(f"{len(has)} orders have BNS section {a.section} as a label\n")

    # courts / file-name patterns, to spot a single template or source
    print("Example file names:")
    for f in has.source_file.head(8):
        print("  ", f[:90])
    print()

    pat = re.compile(rf".{{0,120}}(?<![\d./]){re.escape(a.section)}.{{0,60}}", re.S)
    for _, r in has.head(a.n).iterrows():
        print("=" * 70)
        print(r.source_file[:90], "| all sections:", r.sections)
        for m in list(pat.finditer(r.text))[:2]:
            print("   ...", re.sub(r"\s+", " ", m.group()), "...")


if __name__ == "__main__":
    main()