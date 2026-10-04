#!/usr/bin/env python3
"""
Step 1: PDFs -> clean, de-duplicated CSV with BNS section labels.

Usage:
    python build_dataset.py --pdf_dir pdfs --out_dir out

What it does
  1. Extracts text from every PDF (one big PDF of ~1000 orders is fine).
  2. Splits bundled PDFs into individual judgments (Allahabad HC style:
     each order starts with "Neutral Citation No.").
  3. Removes duplicates: exact (same normalized text / same neutral citation)
     and near-duplicates (5-word shingle Jaccard >= --dup_threshold).
  4. Extracts BNS sections mentioned in each judgment (BNSS is ignored).
  5. Writes text_masked, where the section mentions are replaced by
     [SECTION], so the model has to learn from facts instead of just
     reading the answer out of the text.
  6. Saves out/dataset.csv, out/duplicates_report.csv, out/label_counts.csv
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    sys.exit("pip install pypdf pandas")

# --------------------------------------------------------------------------
# Section extraction
# --------------------------------------------------------------------------
# "BNS" but never "BNSS" (that is the procedure code, not an offence).
BNS_TOKEN = r"(?:B\.?\s?N\.?\s?S\.?|Bharatiya\s+Nyaya\s+Sanhita(?:,?\s*2023)?)(?![A-Za-z])"
BNSS_GUARD = r"(?!\s?S\b)"

SEC_NUM = r"\d{1,3}[A-Za-z]?(?:\s?\(\s?\w{1,4}\s?\))*"

# Old code (IPC). Orders often write "(erstwhile 149 IPC)" next to the BNS
# section, which would leak the answer, so IPC mentions are masked as well.
IPC_TOKEN = r"(?:I\.?\s?P\.?\s?C\.?|Indian\s+Penal\s+Code(?:,?\s*1860)?)(?![A-Za-z])"


def _patterns(token: str, guard: str = ""):
    # Form A: every number directly followed by the code -> "333 BNS", "316(2) BNS"
    direct = re.compile(
        rf"(?<![\d./])({SEC_NUM})\s*(?:of\s+(?:the\s+)?)?{token}{guard}", re.IGNORECASE)
    # Form B: "sections 316(2), 318(4) and 333 of BNS"
    group = re.compile(
        rf"(?:sections?|secs?\.?|ss\.?|u/s\.?)\s*"
        rf"((?:{SEC_NUM}\s*(?:,|and|&|/|r/w|read\s+with|with)\s*)*{SEC_NUM})"
        rf"\s*(?:of\s+(?:the\s+)?)?{token}{guard}", re.IGNORECASE)
    return direct, group


DIRECT_RE, GROUP_RE = _patterns(BNS_TOKEN, BNSS_GUARD)
IPC_DIRECT_RE, IPC_GROUP_RE = _patterns(IPC_TOKEN)
NUM_IN_GROUP = re.compile(r"(\d{1,3})[A-Za-z]?((?:\s?\(\s?\w{1,4}\s?\))*)")


def _clean_sec(num: str, sub: str, keep_sub: bool) -> str | None:
    base = int(re.match(r"\d+", num).group())
    if not 1 <= base <= 358:  # BNS has 358 sections
        return None
    suffix = re.match(r"\d+([A-Za-z]?)", num).group(1).upper()
    out = f"{base}{suffix}"
    if keep_sub and sub:
        out += re.sub(r"\s", "", sub)
    return out


def find_mentions(text: str, keep_sub: bool = False) -> list[tuple[str, int]]:
    """Every BNS section mention as (section, position in text)."""
    out = []
    for m in DIRECT_RE.finditer(text):
        n = NUM_IN_GROUP.match(m.group(1))
        if n:
            s = _clean_sec(n.group(1), n.group(2), keep_sub)
            if s:
                out.append((s, m.start(1)))
    for m in GROUP_RE.finditer(text):
        for n in NUM_IN_GROUP.finditer(m.group(1)):
            s = _clean_sec(n.group(1), n.group(2), keep_sub)
            if s:
                out.append((s, m.start(1) + n.start()))
    return out


def _sort_key(x: str):
    return (int(re.match(r"\d+", x).group()), x)


def extract_sections(text: str, keep_sub: bool = False) -> list[str]:
    return sorted({s for s, _ in find_mentions(text, keep_sub)}, key=_sort_key)


# ---- boilerplate filter -------------------------------------------------
# Courts copy the same bail-condition paragraph into hundreds of orders
# ("if he absconds ... proceed against him under Section 269 BNS"). Those
# sections are NOT the charges of the case, so they must not become labels.
CUE_RE = re.compile(
    r"abscond|absence|absent|fresh\s+FIR|misuse|proceed\s+against\s+(him|her|them)|"
    r"fail(s|ure)?\s+to\s+appear|non-?appearance|jump(s|ed)?\s+bail", re.IGNORECASE)
# a real charge line is tied to a specific case (FIR / crime number / police station)
CASE_GUARD_RE = re.compile(
    r"\bFIR\b|F\.I\.R|crime|registered|police\s+station|\bP\.?S\.?\b|charge|case\s+no|"
    r"alleg|punishable|booked",
    re.IGNORECASE)


def _ctx(text: str, pos: int, n: int) -> str:
    return re.sub(r"\s+", " ", text[max(0, pos - n):pos]).strip().lower()


def label_documents(texts: list[str], keep_sub: bool = False, boiler_threshold: int = 6,
                    drop_sections: tuple = (), drop_ratio: float = 0.6):
    """Return (labels per document, dropped section info, flagged-window report)."""
    from collections import Counter, defaultdict

    mentions = [find_mentions(t, keep_sub) for t in texts]
    win_docs = Counter()
    for t, ms in zip(texts, mentions):
        for w in {_ctx(t, p, 70) for _, p in ms}:
            win_docs[w] += 1

    def flagged(t, p):
        if CUE_RE.search(_ctx(t, p, 150)):
            return "cue"
        w = _ctx(t, p, 70)
        if win_docs[w] >= boiler_threshold and not CASE_GUARD_RE.search(w):
            return "repeated"
        return None

    kept_per_doc, mention_docs, boiler_docs = [], Counter(), Counter()
    report = defaultdict(lambda: [0, ""])  # window -> [docs, section]
    for t, ms in zip(texts, mentions):
        good, bad = set(), set()
        for s, p in ms:
            why = flagged(t, p)
            if why:
                bad.add(s)
                w = _ctx(t, p, 70)
                report[w][0] = win_docs[w]; report[w][1] = s
            else:
                good.add(s)
        for s in good | bad:
            mention_docs[s] += 1
            if s in bad and s not in good:
                boiler_docs[s] += 1
        kept_per_doc.append(good)

    dropped = {}
    for s, n in mention_docs.items():
        ratio = boiler_docs[s] / n
        if (n >= 5 and ratio >= drop_ratio) or s in drop_sections:
            dropped[s] = (n, boiler_docs[s])
    labels = [sorted(g - set(dropped), key=_sort_key) for g in kept_per_doc]
    rep = sorted(([w, c, s] for w, (c, s) in report.items()), key=lambda r: -r[1])
    return labels, dropped, rep


def mask_sections(text: str) -> str:
    for rx in (GROUP_RE, DIRECT_RE, IPC_GROUP_RE, IPC_DIRECT_RE):
        text = rx.sub(" [SECTION] ", text)
    return re.sub(r"\s+", " ", text).strip()


# Where the facts start. Orders begin with court headers, counsel names, case
# numbers... that would eat the model's 512-token window.
FACT_START_RE = re.compile(
    r"\b(F\.?I\.?R\.?|First\s+Information\s+Report|Crime\s+No|Case\s+Crime|"
    r"allegations?|alleged|prosecution|complainant|informant)\b", re.IGNORECASE)
# Several petitions decided together ("CRL.P 552/2026 C/W CRL.P 506/2026 ...").
# NOTE: "X And 2 Others" is just several accused in ONE case, so it is not used.
COMBINED_RE = re.compile(r"\bC/W\b", re.IGNORECASE)


def trim_header(masked: str, back: int = 200) -> str:
    m = FACT_START_RE.search(masked)
    if not m:
        return masked
    start = max(0, m.start() - back)
    if start > 0:  # don't start in the middle of a word
        nxt = masked.find(" ", start)
        start = nxt + 1 if 0 <= nxt < m.start() else start
    return masked[start:]


# --------------------------------------------------------------------------
# PDF -> judgments
# --------------------------------------------------------------------------
SPLIT_RE = re.compile(r"(?=Neutral\s+Citation\s+No\.)", re.IGNORECASE)
CITATION_RE = re.compile(r"Neutral\s+Citation\s+No\.\s*-?\s*([\w:\-]+)", re.IGNORECASE)
CASE_RE = re.compile(r"Case\s*:-\s*(.+)")
DATE_RE = re.compile(r"Order\s+Date\s*:-\s*([\d./\-]+)", re.IGNORECASE)


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def split_judgments(full_text: str) -> list[str]:
    parts = [p.strip() for p in SPLIT_RE.split(full_text) if p.strip()]
    # keep only chunks that really start with a header; glue stray text back
    merged: list[str] = []
    for p in parts:
        if CITATION_RE.match(p) or not merged:
            merged.append(p)
        else:
            merged[-1] += "\n" + p
    return merged


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def shingles(text: str, k: int = 5) -> set[int]:
    w = normalize(text).split()
    if len(w) < k:
        return {hash(" ".join(w))}
    return {hash(" ".join(w[i : i + k])) for i in range(len(w) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="pdfs")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--dup_threshold", type=float, default=0.95,
                    help="near-duplicate cut-off; templated bail orders for different "
                         "people are ~0.86-0.89 similar, so keep this high")
    ap.add_argument("--boiler_threshold", type=int, default=6,
                    help="a section mention whose surrounding words appear word-for-word "
                         "in this many orders (and is not tied to an FIR) is boilerplate")
    ap.add_argument("--drop_sections", nargs="*", default=[],
                    help="force-remove sections from the labels, e.g. --drop_sections 269 209")
    ap.add_argument("--keep_subsection", action="store_true",
                    help="label 316(2) separately from 316 (more labels, sparser)")
    ap.add_argument("--min_words", type=int, default=30,
                    help="drop judgments shorter than this (scanned/empty pages)")
    args = ap.parse_args()

    pdf_dir, out_dir = Path(args.pdf_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir}/")

    rows = []
    for pdf in pdfs:
        text = pdf_text(pdf)
        if len(text.split()) < args.min_words:
            print(f"[warn] {pdf.name}: almost no text (scanned PDF? needs OCR)")
            continue
        for i, chunk in enumerate(split_judgments(text)):
            cit = CITATION_RE.search(chunk)
            case = CASE_RE.search(chunk)
            date = DATE_RE.search(chunk)
            rows.append({
                "source_file": pdf.name,
                "chunk_no": i,
                "neutral_citation": cit.group(1) if cit else "",
                "case_title": case.group(1).strip() if case else "",
                "order_date": date.group(1) if date else "",
                "text": chunk,
            })
    print(f"Extracted {len(rows)} judgments from {len(pdfs)} PDF(s)")

    # ---------------- de-duplication ----------------
    kept, dup_log = [], []
    seen_hash, seen_cit, kept_shingles = {}, {}, []
    for r in rows:
        norm = normalize(r["text"])
        h = hashlib.md5(norm.encode()).hexdigest()
        ident = f'{r["source_file"]}#{r["chunk_no"]}'
        if h in seen_hash:
            dup_log.append((ident, seen_hash[h], "exact_text", 1.0)); continue
        if r["neutral_citation"] and r["neutral_citation"] in seen_cit:
            dup_log.append((ident, seen_cit[r["neutral_citation"]], "same_citation", 1.0)); continue
        sh = shingles(r["text"])
        near = None
        for oid, osh in kept_shingles:
            j = jaccard(sh, osh)
            if j >= args.dup_threshold:
                near = (oid, j); break
        if near:
            dup_log.append((ident, near[0], "near_duplicate", round(near[1], 3))); continue
        seen_hash[h] = ident
        if r["neutral_citation"]:
            seen_cit[r["neutral_citation"]] = ident
        kept_shingles.append((ident, sh))
        kept.append(r)
    print(f"Removed {len(dup_log)} duplicates -> {len(kept)} unique judgments")
    pd.DataFrame(dup_log, columns=["dropped", "kept_copy", "reason", "similarity"]) \
        .to_csv(out_dir / "duplicates_report.csv", index=False)

    # ---------------- labels ----------------
    all_labels, dropped, boiler_rep = label_documents(
        [r["text"] for r in kept], keep_sub=args.keep_subsection,
        boiler_threshold=args.boiler_threshold, drop_sections=tuple(args.drop_sections))
    pd.DataFrame(boiler_rep[:60], columns=["text_before_section", "orders", "section"]) \
        .to_csv(out_dir / "boilerplate_report.csv", index=False)
    if dropped:
        print("\nSections removed from the labels because they come from copied "
              "boilerplate, not from the charges:")
        for s, (n, b) in sorted(dropped.items(), key=lambda kv: -kv[1][0]):
            print(f"  BNS {s}: mentioned in {n} orders, {b} of them only inside boilerplate")
    recs = []
    for n, r in enumerate(kept):
        secs = all_labels[n]
        masked = mask_sections(r["text"])
        model_text = trim_header(masked)
        recs.append({
            "doc_id": f"doc{n:05d}",
            **{k: r[k] for k in ("source_file", "neutral_citation", "case_title", "order_date")},
            "sections": "|".join(secs),
            "n_sections": len(secs),
            "n_words": len(masked.split()),
            "is_combined": bool(COMBINED_RE.search(r["text"][:3000])),
            "text_model": model_text,
            "text_masked": masked,
            "text": r["text"],
        })
    df = pd.DataFrame(recs)
    no_label = (df.n_sections == 0).sum()
    df.to_csv(out_dir / "dataset.csv", index=False)

    counts = (df.sections[df.sections != ""].str.split("|").explode()
              .value_counts().rename_axis("bns_section").reset_index(name="count"))
    counts.to_csv(out_dir / "label_counts.csv", index=False)

    print(f"\nSaved {out_dir}/dataset.csv  ({len(df)} rows)")
    print(f"  rows with no BNS section found : {no_label}  (dropped at training time)")
    print(f"  combined multi-petition orders : {int(df.is_combined.sum())}  (skipped at training time)")
    print(f"  distinct BNS sections          : {len(counts)}")
    print(f"  median words after masking     : {int(df.n_words.median()) if len(df) else 0}")
    print("Top sections:\n", counts.head(10).to_string(index=False))


if __name__ == "__main__":
    main()