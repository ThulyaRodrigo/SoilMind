"""Ingestion: raw documents -> data/processed/passages.csv.

    python -m member4_chatbot.ingest.run

Steps per document (see extract.py, clean.py, chunk.py): extract text with the right method, clean it, chunk it.
Outputs:
- data/processed/passages.csv: id, text, lang, doc, page, url, plus doc_title, section, n_words
- data/processed/extraction_log.csv: one row per document with the method used and its checks
"""

from __future__ import annotations

import hashlib

import pandas as pd
import yaml

from collections import Counter

from member4_chatbot.ingest.chunk import chunk_pages, to_blocks
from member4_chatbot.ingest.clean import clean_lines, clean_pdf_pages, site_furniture
from member4_chatbot.ingest.extract import extract
from member4_chatbot.ingest.sinhala import sinhala_share
from member4_chatbot.lang import detect_language
from member4_chatbot.paths import DATA_DIR, PASSAGES_CSV, PROCESSED_DIR, RAW_DOCS_DIR

MIN_PASSAGE_WORDS = 15
MANUAL_DIR = DATA_DIR / "docs" / "manual_text"
PASSAGE_COLUMNS = ["id", "text", "lang", "doc", "page", "url", "doc_title", "title", "section", "n_words"]
MAX_TITLED_PAGES = 6  # single-topic documents (web pages, leaflets); not books
GENERIC_HEADING_DOCS = 6  # a heading used by this many documents or more is a site banner, not a title


def document_titles(cleaned: dict[str, list[tuple[int, str]]]) -> dict[str, str]:
    """Each document's title: its first heading that is not a site-wide banner ("Rice In Sri Lanka - Pests of
    Rice", "RRDI Rice Varieties" appear on many pages; "Brown Plant Hopper", "Bg 300" do not)."""
    headings = {doc: [b.text for b in to_blocks(pages) if b.kind == "heading"] for doc, pages in cleaned.items()}
    used_by = Counter(h.lower() for hs in headings.values() for h in set(hs))
    first_lines = {doc: [ln.strip() for _, t in pages for ln in t.split(chr(10)) if ln.strip()][:3]
                   for doc, pages in cleaned.items()}
    line_used_by = Counter(ln.lower() for lines in first_lines.values() for ln in set(lines))
    titles = {}
    for doc, hs in headings.items():
        ok = [h for h in hs if used_by[h.lower()] < GENERIC_HEADING_DOCS and len(h.split()) <= 12]
        # no usable heading (e.g. technology pages): the first short line that is not a site banner
        ok += [ln for ln in first_lines[doc] if line_used_by[ln.lower()] < GENERIC_HEADING_DOCS
               and len(ln.split()) <= 12 and '|' not in ln]
        # books cover many topics: one title for all their passages would mislead, so they get none
        titles[doc] = ok[0] if ok and len(cleaned[doc]) <= MAX_TITLED_PAGES else ''
    return titles


def with_title(text: str, title: str) -> str:
    """Contextual chunk header: a passage that does not already mention its document's title gets it as the first
    line, so e.g. the resistance table of a variety sheet still says which variety it belongs to."""
    return text if not title or title.lower() in text.lower() else f"{title}\n{text}"


def load_manifest() -> list[dict]:
    m = yaml.safe_load((RAW_DOCS_DIR / "manifest.yaml").read_text(encoding="utf-8"))
    return [d for d in m["documents"] if d.get("status") in ("ok", "manual") and d.get("file")]


def main() -> pd.DataFrame:
    docs = load_manifest()
    extracted = {d["id"]: extract(d, RAW_DOCS_DIR, MANUAL_DIR) for d in docs}
    furniture = site_furniture([e.pages[0][1] for e in extracted.values() if e.method == "html_text"])

    cleaned = {}
    for d in docs:
        ex = extracted[d["id"]]
        if ex.method == "html_text":
            cleaned[d["id"]] = [(p, clean_lines(t, furniture)) for p, t in ex.pages]
        else:
            cleaned[d["id"]] = clean_pdf_pages(ex.pages)
    titles = document_titles(cleaned)

    rows, log, seen = [], [], set()
    for d in docs:
        ex = extracted[d["id"]]
        pages = cleaned[d["id"]]
        chunks = chunk_pages(pages) if pages else []
        kept = 0
        for c in chunks:
            if c.words < MIN_PASSAGE_WORDS:
                continue
            h = hashlib.sha1(c.text.encode("utf-8")).hexdigest()
            if h in seen:  # the same page reached twice (e.g. two URLs)
                continue
            seen.add(h)
            lang = detect_language(c.text)
            text = with_title(c.text, titles[d["id"]])
            rows.append({
                "id": f"{d['id']}__{kept:03d}", "text": text, "lang": "si" if lang == "si" else "en",
                "doc": d["id"], "page": c.page, "url": d["url"], "doc_title": d["title"], "title": titles[d["id"]],
                "section": c.section, "n_words": len(text.split()),
            })
            kept += 1
        full = "\n".join(t for _, t in pages)
        log.append({
            "doc": d["id"], "file": d["file"], "language": d["language"], "method": ex.method,
            "pages": len(ex.pages), "chars": len(full), "sinhala_share": round(sinhala_share(full), 3),
            "passages": kept, "note": ex.note,
            "checks": "; ".join(f"{k}: {v['reason']} (si share {v['sinhala_share']:.2f}, word rate {v['common_word_rate']:.2f})"
                                for k, v in ex.checks.items()),
        })
        print(f"{ex.method:<18} {d['id']:<50} passages={kept:4} {ex.note}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=PASSAGE_COLUMNS)
    df.to_csv(PASSAGES_CSV, index=False, encoding="utf-8")
    pd.DataFrame(log).to_csv(PROCESSED_DIR / "extraction_log.csv", index=False, encoding="utf-8")
    by = df.groupby("lang").size().to_dict()
    print(f"{len(df)} passages ({by}); median {df.n_words.median():.0f} words -> {PASSAGES_CSV}")
    return df


if __name__ == "__main__":
    main()
