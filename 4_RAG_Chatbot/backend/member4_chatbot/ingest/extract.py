"""Text extraction for each raw document, choosing the method per document and logging it.

Order of methods (Phase 5 brief):
1. **manual_text**: a text file the team supplied in data/docs/manual_text/<id>.txt (e.g. after OCR elsewhere).
   It always wins, so a person can fix any document by hand.
2. **html_text**: the `.txt` saved from a doa.gov.lk page (already Unicode).
3. **text_layer**: the PDF's own text (PyMuPDF), when it reads as real Sinhala / English.
4. **legacy_font_map**: the text layer mapped from an FM-style legacy font to Unicode, when that reads as Sinhala.
5. **ocr_tesseract**: Tesseract `sin+eng` on 300-dpi page images.
6. **skipped_needs_ocr**: nothing above worked (e.g. Tesseract is not installed). The document is left out and
   listed in the extraction log, so the user can install Tesseract or supply manual text.
"""

from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from member4_chatbot.ingest.sinhala import assess, fm_to_unicode

OCR_DPI = 300
OCR_LANG = "sin+eng"


@dataclass
class Extracted:
    doc_id: str
    method: str
    pages: list[tuple[int, str]] = field(default_factory=list)  # (1-based page, text)
    note: str = ""
    checks: dict = field(default_factory=dict)


def tesseract_cmd() -> str | None:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return str(default) if default.exists() else None


def tesseract_has_sinhala(cmd: str) -> bool:
    import subprocess

    try:
        out = subprocess.run([cmd, "--list-langs"], capture_output=True, text=True, timeout=30)
    except OSError:
        return False
    return "sin" in out.stdout.split()


def ocr_images(images: list[tuple[int, bytes]]) -> list[tuple[int, str]]:
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd()
    return [(p, pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=OCR_LANG)) for p, png in images]


def page_blocks_text(page) -> str:
    """A PDF page as paragraphs: one PyMuPDF text block per paragraph, wrapped lines joined."""
    paras = []
    for b in page.get_text("blocks", sort=True):
        if b[6] != 0:  # image block
            continue
        lines = [ln.strip() for ln in b[4].splitlines() if ln.strip()]
        text = ""
        for ln in lines:
            if text.endswith("-") and ln[:1].islower():
                text = text[:-1] + ln  # re-join a hyphenated word
            else:
                text = f"{text} {ln}" if text else ln
        if text:
            paras.append(text)
    return "\n\n".join(paras)


def pdf_page_images(path: Path) -> list[tuple[int, bytes]]:
    doc = pymupdf.open(path)
    return [(i + 1, page.get_pixmap(dpi=OCR_DPI).tobytes("png")) for i, page in enumerate(doc)]


def extract(doc: dict, raw_dir: Path, manual_dir: Path) -> Extracted:
    """`doc` is a manifest entry (id, file, language)."""
    doc_id, lang = doc["id"], doc["language"]
    manual = manual_dir / f"{doc_id}.txt"
    if manual.exists():
        pages = [(i + 1, t) for i, t in enumerate(manual.read_text(encoding="utf-8").split("\f"))]
        return Extracted(doc_id, "manual_text", pages, note=str(manual.name))

    path = raw_dir / doc["file"]
    if path.suffix == ".txt":
        return Extracted(doc_id, "html_text", [(1, path.read_text(encoding="utf-8"))])

    checks: dict = {}
    if path.suffix == ".pdf":
        pdf = pymupdf.open(path)
        pages = [(i + 1, page_blocks_text(p)) for i, p in enumerate(pdf)]
        full = "\n".join(t for _, t in pages)
        a = assess(full, lang)
        checks["text_layer"] = a
        if a["ok"]:
            return Extracted(doc_id, "text_layer", pages, checks=checks)
        if a["reason"] != "no text layer":
            mapped = [(p, fm_to_unicode(t)) for p, t in pages]
            m = assess("\n".join(t for _, t in mapped), "si")
            checks["legacy_font_map"] = m
            if m["ok"]:
                return Extracted(doc_id, "legacy_font_map", mapped, checks=checks)
        images = None
    else:  # .jpg / .png
        images = [(1, path.read_bytes())]

    cmd = tesseract_cmd()
    if cmd and tesseract_has_sinhala(cmd):
        ocr_pages = ocr_images(images if images is not None else pdf_page_images(path))
        o = assess("\n".join(t for _, t in ocr_pages), lang)
        checks["ocr_tesseract"] = o
        return Extracted(doc_id, "ocr_tesseract", ocr_pages, checks=checks,
                         note="" if o["ok"] else "OCR output did not pass the readability check")
    why = "Tesseract not installed" if not cmd else "Tesseract has no 'sin' language data"
    return Extracted(doc_id, "skipped_needs_ocr", [], note=why, checks=checks)
