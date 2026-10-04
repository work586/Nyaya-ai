#!/usr/bin/env python3
"""
BNS Charge Prediction System - backend.

    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:5000

Flow: register/login -> upload judgment PDF -> extract + clean text -> judgment details
      -> multi-label BNS prediction (confidence + evidence) -> saved to history -> PDF report.

Needs the saved model first:  python baseline_tfidf.py --min_label_count 30 --save model_tfidf
"""
import io
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from xml.sax.saxutils import escape

import joblib
import numpy as np
from flask import Flask, Response, g, jsonify, request, send_file, session
from pypdf import PdfReader
from werkzeug.security import check_password_hash, generate_password_hash

# text helpers shared with the dataset builder, so uploads are treated exactly like training data
from build_dataset import CUE_RE, COMBINED_RE, FACT_START_RE, _ctx, find_mentions, mask_sections, trim_header

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model_tfidf" / "model.joblib"
DB_PATH = BASE / "bns_app.db"
SECRET_PATH = BASE / "secret.key"

# Short names, for display only. Please check against the official BNS text before presenting.
TITLES = {
    "3": "General explanations (3(5): common intention)", "61": "Criminal conspiracy",
    "103": "Murder", "105": "Culpable homicide not amounting to murder",
    "109": "Attempt to murder", "110": "Attempt to commit culpable homicide",
    "115": "Voluntarily causing hurt", "117": "Voluntarily causing grievous hurt",
    "118": "Hurt by dangerous weapons or means", "126": "Wrongful restraint",
    "189": "Unlawful assembly", "190": "Liability of members of an unlawful assembly",
    "191": "Rioting", "238": "Causing disappearance of evidence", "296": "Obscene acts and songs",
    "303": "Theft", "316": "Criminal breach of trust", "318": "Cheating",
    "319": "Cheating by personation", "324": "Mischief",
    "333": "House-trespass after preparation for hurt or restraint", "336": "Forgery",
    "338": "Forgery of valuable security, will, etc.", "340": "Using a forged document as genuine",
    "351": "Criminal intimidation", "352": "Insult intended to provoke breach of the peace",
}

if not MODEL_PATH.exists():
    sys.exit(f"Model not found at {MODEL_PATH}.\n"
             "Create it first:  python baseline_tfidf.py --min_label_count 30 --save model_tfidf")
bundle = joblib.load(MODEL_PATH)
vec, clf = bundle["vec"], bundle["clf"]
labels, threshold = bundle["labels"], float(bundle["threshold"])
metrics = bundle.get("metrics")
terms = vec.get_feature_names_out()

# ----------------------------------------------------------------------------- app
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
if not SECRET_PATH.exists():
    SECRET_PATH.write_text(os.urandom(32).hex())
app.config.update(
    SECRET_KEY=SECRET_PATH.read_text().strip(),
    SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=25 * 1024 * 1024,
)


def err(msg, code=400):
    return jsonify(error=msg), code


@app.errorhandler(413)
def too_big(_):
    return err("That file is too large (limit 25 MB).", 413)


@app.before_request
def csrf_guard():
    # state-changing API calls must come from our own page (it sends this header)
    if request.path.startswith("/api/") and request.method in ("POST", "DELETE"):
        if request.headers.get("X-Requested-With") != "fetch":
            return err("Bad request.", 403)


# ----------------------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
  pw_hash TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analyses(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id),
  created_at TEXT NOT NULL, filename TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id, id DESC);
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d is not None:
        d.close()


with sqlite3.connect(DB_PATH) as _c:
    _c.executescript(SCHEMA)


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ----------------------------------------------------------------------------- auth
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
FAILS: dict = {}  # (ip, email) -> [timestamps]  (very small brute-force limiter)


def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if "uid" not in session:
            return err("Please log in.", 401)
        return f(*a, **k)
    return wrapper


def current_user():
    if "uid" not in session:
        return None
    r = db().execute("SELECT id, name, email FROM users WHERE id=?", (session["uid"],)).fetchone()
    if r is None:
        session.clear()
        return None
    return {"id": r["id"], "name": r["name"], "email": r["email"]}


@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    name = str(d.get("name", "")).strip()
    email = str(d.get("email", "")).strip().lower()
    pw = str(d.get("password", ""))
    if not 1 <= len(name) <= 60:
        return err("Please enter your name (up to 60 characters).")
    if not EMAIL_RE.match(email):
        return err("Please enter a valid email address.")
    if len(pw) < 8:
        return err("Password must be at least 8 characters.")
    if db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        return err("An account with this email already exists.", 409)
    cur = db().execute("INSERT INTO users(name,email,pw_hash,created_at) VALUES(?,?,?,?)",
                       (name, email, generate_password_hash(pw), now()))
    db().commit()
    session.clear(); session.permanent = True; session["uid"] = cur.lastrowid
    return jsonify(user=current_user())


@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    email = str(d.get("email", "")).strip().lower()
    pw = str(d.get("password", ""))
    key = (request.remote_addr, email)
    recent = [t for t in FAILS.get(key, []) if time.time() - t < 300]
    if len(recent) >= 5:
        return err("Too many failed attempts. Please wait a few minutes and try again.", 429)
    r = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if r is None or not check_password_hash(r["pw_hash"], pw):
        FAILS[key] = recent + [time.time()]
        return err("Incorrect email or password.", 401)
    FAILS.pop(key, None)
    session.clear(); session.permanent = True; session["uid"] = r["id"]
    return jsonify(user=current_user())


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/me")
def me():
    return jsonify(user=current_user())


# ----------------------------------------------------------------------------- PDF -> details
COURT_CODES = {
    "AHC-LKO": "Allahabad High Court, Lucknow Bench", "AHC": "Allahabad High Court",
    "UHC": "Uttarakhand High Court", "KHC": "Karnataka High Court", "DHC": "Delhi High Court",
    "BHC": "Bombay High Court", "PHHC": "Punjab and Haryana High Court", "GUJHC": "Gujarat High Court",
}
NEUTRAL_RE = re.compile(r"\b((?:19|20)\d{2}\s?:\s?([A-Z]{2,6}(?:-[A-Z]{2,4})?)\s?:\s?\d{1,7})\b")
COURT_RE = re.compile(
    r"(SUPREME\s+COURT\s+OF\s+INDIA"
    r"|HIGH\s+COURT\s+OF\s+JUDICATURE\s+(?:AT|FOR)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?"
    r"|HIGH\s+COURT\s+(?:OF|FOR)\s+[A-Za-z][A-Za-z&\s]{2,50}?"
    r"(?=\s+(?:AT|BENCH|BEFORE|CORAM|CRM|CRL|CRIMINAL|BAIL|DATED|DATE|PRESENT|NO\b|PETITION|"
    r"APPLICATION|WRIT|HABEAS|\d)|\s*$|[.,:;(]))", re.I)
CASE_LINE_RE = re.compile(r"Case\s*:-\s*(.+?)(?=\s+(?:Applicant|Petitioner|Appellant|Counsel)\b|$)", re.I | re.S)
CASE_NUM_RE = re.compile(r"\bNos?\.?\s*[-:]?\s*(\d{1,6})\s*(?:of|/)\s*((?:19|20)\d{2})", re.I)
TYPE_WORDS = {"criminal", "crl", "crl.", "cr", "cr.", "crm", "misc", "misc.", "miscellaneous", "bail",
              "application", "petition", "appeal", "revision", "writ", "habeas", "corpus", "original",
              "anticipatory", "suo", "motu"}
SKIP_TYPES = {"fir", "f.i.r", "f.i.r.", "crime"}
CASE_NO2_RE = re.compile(r"\b([A-Z]{1,6}\d?/\d{1,6}/(?:19|20)\d{2})\b")
VERSUS_RE = re.compile(r"\s(?:versus|vs\.?|v/s)\s", re.I)
ROLE_RE = re.compile(r"[\s.…\-_,]*\b(?:petitioners?|applicants?|appellants?|respondents?|"
                     r"opposite\s+party|opp\.?\s*party|accused)\b.*$", re.I)
DATE_RES = [
    re.compile(r"Order\s+Date\s*:-\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.I),
    re.compile(r"Date\s+of\s+Decision\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.I),
    re.compile(r"DATED\s+THIS\s+THE\s+(\d{1,2})\s*(?:ST|ND|RD|TH)?\s*DAY\s+OF\s+([A-Z]+),?\s*(\d{4})", re.I),
    re.compile(r"pronounced[^.]{0,60}?(\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2})", re.I),
    re.compile(r"Dated\s*:\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", re.I),
    re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2})\b"),
]
SENT_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(\[0-9\"'])")
ISSUE_RE = re.compile(r"\b(submit(?:s|ted)?\s+that|contend(?:s|ed)?\s+that|argu(?:e|es|ed)\s+that|whether|"
                      r"the\s+(?:issue|question)|(?:issue|question)\s+(?:for|before))\b", re.I)
ORDER_RE = re.compile(r"\b(allowed|dismissed|disposed|granted|rejected|quashed|set\s+aside|released|"
                      r"directed|ordered|bail)\b", re.I)


def smart_title(s):
    s = s.title()
    for w in ("Of", "And", "At", "For", "The"):
        s = re.sub(rf"\b{w}\b", w.lower(), s)
    return s


def pdf_to_text(data: bytes):
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("This PDF is password protected.")
        n_pages = len(reader.pages)
        if n_pages > 200:
            raise ValueError("This PDF has more than 200 pages. Please upload a single judgment.")
        parts = [(p.extract_text() or "") for p in reader.pages]
    except ValueError:
        raise
    except Exception:
        raise ValueError("Could not read this PDF. It may be damaged.")
    return "\n".join(parts), n_pages


def clean_text(raw: str) -> str:
    t = raw.replace(" ", " ").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # re-join words split at line ends
    t = re.sub(r"[ \t​]+", " ", t)
    return re.sub(r"\s*\n\s*", " ", t).strip()       # one flat string


def sentences_of(flat: str, limit=3000):
    return [s.strip() for s in SENT_SPLIT.split(flat) if len(s.strip()) > 1][:limit]


def clip(s, n):
    s = s.strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."


def extract_details(flat: str, n_pages: int) -> dict:
    head = flat[:3000]
    neutral = NEUTRAL_RE.search(head)
    citation, code = (re.sub(r"\s", "", neutral.group(1)), neutral.group(2)) if neutral else ("", "")

    court = ""
    m = COURT_RE.search(head)
    if m:
        court = smart_title(re.sub(r"\s+", " ", m.group(1)).strip())
    elif code in COURT_CODES:
        court = COURT_CODES[code]
    elif "AHC-LKO" in head:
        court = COURT_CODES["AHC-LKO"]

    case_no = ""
    m = CASE_LINE_RE.search(head)
    if m:
        case_no = re.sub(r"\s+", " ", m.group(1)).strip()[:90]
    if not case_no:
        for m in CASE_NUM_RE.finditer(head[:2500]):
            toks = head[max(0, m.start() - 60):m.start()].split()
            if not toks or toks[-1].lower() in SKIP_TYPES:
                continue
            out = [toks.pop()]
            while toks and toks[-1].lower() in TYPE_WORDS and len(out) < 4:
                out.insert(0, toks.pop())
            case_no = f"{' '.join(out)} No. {m.group(1)} of {m.group(2)}"
            break
    if not case_no:
        m = CASE_NO2_RE.search(head)
        case_no = m.group(1) if m else ""

    title = ""
    m = VERSUS_RE.search(head)
    if m:
        left = head[max(0, m.start() - 110):m.start()]
        right = head[m.end():m.end() + 90]
        left = re.split(r"(?:19|20)\d{2}\)?\s+", left)[-1]
        left, right = ROLE_RE.sub("", left).strip(" .-"), ROLE_RE.sub("", right).strip(" .-")
        if 2 < len(left) < 90 and 2 < len(right) < 90:
            title = f"{left} v. {right}"
    if not title:
        m = re.search(r"(?:Applicant|Petitioner)s?\s*:-\s*(.+?)\s+(?:Opposite\s+Party|Respondent)s?\s*:-\s*(.+?)\s+"
                      r"(?:Counsel|Hon)", head, re.I)
        if m:
            title = f"{m.group(1).strip()} v. {m.group(2).strip()}"[:160]

    date, scope = "", head + " " + flat[-1500:]
    for rx in DATE_RES:
        m = rx.search(scope)
        if m:
            date = " ".join(m.groups()) if rx is DATE_RES[2] else m.group(1)
            break

    sents = sentences_of(flat)
    # facts: from the first mention of the FIR / allegations onwards
    start = next((i for i, x in enumerate(sents) if FACT_START_RE.search(x)), 0)
    facts = clip(" ".join(sents[start:start + 6]), 900)
    # contentions: sentences where the arguments or the question for the court are stated
    issues = [clip(s, 300) for s in sents if ISSUE_RE.search(s) and 40 < len(s) < 900][:3]
    # final order: the last sentences that contain an operative word
    tail = sents[int(len(sents) * 0.7):] or sents
    ops = [s for s in tail if ORDER_RE.search(s) and len(s) > 15]
    final = " ".join(clip(s, 300) for s in ops[-2:]) if ops else clip(flat[-400:], 400)

    return {"title": title, "court": court, "case_number": case_no, "citation": citation,
            "date": date, "pages": n_pages, "facts": facts, "issues": issues, "final_order": final}


# ----------------------------------------------------------------------------- prediction
def keywords_for(x, i, k=5):
    est = clf.estimators_[i]
    if not hasattr(est, "coef_"):
        return []
    row = x.multiply(est.coef_[0]).tocsr()
    best = np.argsort(-row.data)[:k]
    return [str(terms[row.indices[j]]) for j in best if row.data[j] > 0]


def evidence_for(sents, X_sent, i, k=2):
    """The sentences of the judgment that push this section's score up the most."""
    est = clf.estimators_[i]
    if not hasattr(est, "coef_") or X_sent is None:
        return []
    scores = np.asarray(X_sent.dot(est.coef_[0])).ravel()
    out = []
    for j in np.argsort(-scores):
        if scores[j] <= 0:
            break
        if len(sents[j]) >= 40:
            out.append(clip(sents[j], 320))
            if len(out) == k:
                break
    return out


def cited_sections(flat: str):
    secs = set()
    for s, pos in find_mentions(flat):
        if not CUE_RE.search(_ctx(flat, pos, 150)):   # skip bail-condition boilerplate
            secs.add(s)
    key = lambda s: (int(re.match(r"\d+", s).group()), s)
    return [{"section": s, "title": TITLES.get(s, ""), "in_model": s in labels} for s in sorted(secs, key=key)]


def run_analysis(data: bytes, filename: str) -> dict:
    raw, n_pages = pdf_to_text(data)
    flat = clean_text(raw)
    n_words = len(flat.split())
    if n_words < 40:
        raise ValueError("No readable text was found. This looks like a scanned PDF, which is not supported yet.")
    flat = " ".join(flat.split()[:60000])

    details = extract_details(flat, n_pages)
    sents = sentences_of(flat)
    model_text = trim_header(mask_sections(flat))        # same preprocessing as the training data
    x = vec.transform([model_text])
    if x.nnz == 0:
        raise ValueError("None of the words in this document were recognised by the model.")
    p = clf.predict_proba(x)[0]
    X_sent = vec.transform([mask_sections(s) for s in sents]) if sents else None

    preds = []
    for rank, i in enumerate(np.argsort(-p)[:8]):
        s, yes = labels[i], bool(p[i] >= threshold)
        show = yes or rank < 3
        preds.append({"section": s, "title": TITLES.get(s, ""), "confidence": round(float(p[i]), 3),
                      "predicted": yes, "keywords": keywords_for(x, i) if show else [],
                      "evidence": evidence_for(sents, X_sent, i) if show else []})

    warnings = []
    if n_words < 150:
        warnings.append("This document is very short, so the prediction is based on little text.")
    if COMBINED_RE.search(flat[:3000]):
        warnings.append("This looks like several petitions decided together; predictions may be unreliable.")
    return {"filename": filename, "n_words": n_words, "details": details, "predictions": preds,
            "threshold": threshold, "cited": cited_sections(flat), "warnings": warnings}


# ----------------------------------------------------------------------------- analyze + history
def row_to_analysis(r):
    a = json.loads(r["data"])
    a["id"], a["created_at"] = r["id"], r["created_at"]
    return a


@app.post("/api/analyze")
@login_required
def analyze():
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("Please choose a PDF file.")
    data = f.read()
    if not data.startswith(b"%PDF-"):
        return err("That file is not a PDF.")
    name = os.path.basename(f.filename.replace("\\", "/"))[:120] or "judgment.pdf"
    try:
        result = run_analysis(data, name)
    except ValueError as e:
        return err(str(e))
    created = now()
    cur = db().execute("INSERT INTO analyses(user_id,created_at,filename,data) VALUES(?,?,?,?)",
                       (session["uid"], created, name, json.dumps(result)))
    db().commit()
    result["id"], result["created_at"] = cur.lastrowid, created
    return jsonify(result)


@app.get("/api/history")
@login_required
def history():
    rows = db().execute("SELECT * FROM analyses WHERE user_id=? ORDER BY id DESC LIMIT 200",
                        (session["uid"],)).fetchall()
    out = []
    for r in rows:
        a = row_to_analysis(r)
        out.append({"id": a["id"], "created_at": a["created_at"], "filename": a["filename"],
                    "title": a["details"].get("title", ""), "court": a["details"].get("court", ""),
                    "predicted": [p["section"] for p in a["predictions"] if p["predicted"]]})
    return jsonify(items=out)


def owned(aid):
    return db().execute("SELECT * FROM analyses WHERE id=? AND user_id=?", (aid, session["uid"])).fetchone()


@app.get("/api/history/<int:aid>")
@login_required
def history_item(aid):
    r = owned(aid)
    return jsonify(row_to_analysis(r)) if r else err("Not found.", 404)


@app.delete("/api/history/<int:aid>")
@login_required
def history_delete(aid):
    if not owned(aid):
        return err("Not found.", 404)
    db().execute("DELETE FROM analyses WHERE id=? AND user_id=?", (aid, session["uid"]))
    db().commit()
    return jsonify(ok=True)


# ----------------------------------------------------------------------------- report
def rl(s):
    """Escape for ReportLab and drop characters the built-in fonts cannot draw (e.g. Devanagari)."""
    return escape(str(s)).encode("latin-1", "replace").decode("latin-1")


def build_report_pdf(a: dict, user_name: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=colors.HexColor("#555555"))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4,
                        textColor=colors.HexColor("#1f4e79"))
    d = a["details"]
    grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6"))])

    story = [Paragraph("BNS Charge Prediction Report", h1),
             Paragraph(rl(f"File: {a['filename']} | Analysed: {a['created_at']} | User: {user_name}"), small),
             Paragraph("Judgment details (auto-extracted)", h2)]
    rows = [[Paragraph("<b>Field</b>", body), Paragraph("<b>Value</b>", body)]]
    for k, v in (("Title", d.get("title")), ("Court", d.get("court")), ("Case number", d.get("case_number")),
                 ("Neutral citation", d.get("citation")), ("Date", d.get("date")), ("Pages", d.get("pages"))):
        rows.append([Paragraph(k, body), Paragraph(rl(v) if v else "Not found", body)])
    t = Table(rows, colWidths=[40 * mm, 130 * mm]); t.setStyle(grid); story.append(t)

    for head, text in (("Facts", d.get("facts")), ("Issues / contentions", "  ".join(d.get("issues") or [])),
                       ("Final order", d.get("final_order"))):
        story += [Paragraph(head, h2), Paragraph(rl(text) if text else "Not found", body)]

    story.append(Paragraph("Predicted BNS sections", h2))
    rows = [[Paragraph(f"<b>{h}</b>", body) for h in ("Section", "Description", "Confidence", "Result")]]
    for p in a["predictions"]:
        rows.append([Paragraph(f"BNS {rl(p['section'])}", body), Paragraph(rl(p["title"] or "-"), body),
                     Paragraph(f"{round(p['confidence'] * 100)}%", body),
                     Paragraph("Predicted" if p["predicted"] else "Below threshold", body)])
    t = Table(rows, colWidths=[24 * mm, 82 * mm, 24 * mm, 40 * mm]); t.setStyle(grid); story.append(t)
    story.append(Paragraph(f"Decision threshold: {round(a['threshold'] * 100)}%. Confidence is the model's score, "
                           "not a calibrated probability.", small))

    ev = [p for p in a["predictions"] if p["predicted"] and (p["evidence"] or p["keywords"])]
    if ev:
        story.append(Paragraph("Evidence from the judgment", h2))
        for p in ev:
            story.append(Paragraph(f"<b>BNS {rl(p['section'])}</b>"
                                   + (f" - keywords: {rl(', '.join(p['keywords']))}" if p["keywords"] else ""), body))
            for s in p["evidence"]:
                story.append(Paragraph("&bull; " + rl(s), body))
            story.append(Spacer(1, 4))

    if a.get("cited"):
        story.append(Paragraph("BNS sections cited in the document (for comparison)", h2))
        story.append(Paragraph(rl(", ".join(c["section"] for c in a["cited"])), body))
    for w in a.get("warnings", []):
        story.append(Paragraph("Note: " + rl(w), small))

    story += [Spacer(1, 10), Paragraph(
        "This report is produced by a machine-learning model trained on a small set of High Court bail orders. "
        "It is decision support for a student project, is not legal advice, and can be wrong.", small)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm, title="BNS Charge Prediction Report").build(story)
    return buf.getvalue()


@app.get("/api/history/<int:aid>/report.pdf")
@login_required
def report(aid):
    r = owned(aid)
    if not r:
        return err("Not found.", 404)
    try:
        pdf = build_report_pdf(row_to_analysis(r), current_user()["name"])
    except ImportError:
        return err("Reports need the reportlab package:  pip install reportlab", 500)
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="bns_report_{aid}.pdf"'})


# ----------------------------------------------------------------------------- public pages
@app.get("/api/info")
def info():
    return jsonify(n_labels=len(labels), threshold=threshold, metrics=metrics,
                   sections=[{"section": s, "title": TITLES.get(s, "")} for s in labels])


@app.get("/")
def home():
    page = BASE / "index.html"
    if not page.exists():
        page = BASE / "static" / "index.html"
    return send_file(page)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)