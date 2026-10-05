"""Chunking: split a cleaned document into passages of about 150-300 words.

Rules:
- A document is a list of blocks: headings (`## ...`, or a short title-like line), tables (`[TABLE] ... [/TABLE]`)
  and paragraphs.
- A heading starts a new passage (unless the current one is still very short) and is kept at the top of the
  passage, so each passage says what it is about.
- A table is never split. A table longer than 300 words becomes one passage on its own.
- Paragraphs are packed until the next one would pass 300 words. A paragraph longer than 300 words is split at
  sentence ends.
- A paragraph or table repeated within the same document is kept once.
- A last passage under 50 words is merged into the one before it.
Words are counted by whitespace, which works for both English and Sinhala.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_WORDS, MAX_WORDS, TAIL_WORDS = 150, 300, 50
SENTENCE_END = re.compile(r"(?<=[.!?෴])\s+")


@dataclass
class Block:
    kind: str  # heading | table | para
    text: str
    page: int

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Chunk:
    text: str
    page: int
    section: str

    @property
    def words(self) -> int:
        return len(self.text.split())


VARIETY_CODE = re.compile(r"^(?:[A-Z][a-z]{0,2}\s*)?\d{1,3}(?:\s*[-–/]\s*\d+[a-z]?)?(?:\s*\([^)]*\))?$")  # "Bg 300", "62-355"


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if VARIETY_CODE.match(s) and re.search(r"[A-Za-z]", s) or re.fullmatch(r"\d{2}\s*[-–]\s*\d{3}", s):
        return True  # a rice variety code on its own line starts that variety's description (book B2)
    if not s or len(s.split()) > 10 or s.endswith((".", ",", ":", ";")) or "|" in s:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    if all(c.isascii() for c in letters):
        return s.isupper() and len(letters) > 3 or (s[0].isupper() and s.istitle() and len(s.split()) >= 2)
    return False  # Sinhala has no case; only explicit `##` headings count


def to_blocks(pages: list[tuple[int, str]]) -> list[Block]:
    """Blocks of one document. A paragraph or table that already appeared earlier in the same document is
    dropped (doa.gov.lk pages often repeat a section in tabs), as is "Lorem ipsum" placeholder text."""
    blocks: list[Block] = []
    for page, text in pages:
        lines = text.split("\n")
        i = 0
        para: list[str] = []

        def flush_para():
            if para:
                blocks.append(Block("para", " ".join(para), page))
                para.clear()

        while i < len(lines):
            ln = lines[i].strip()
            if ln == "[TABLE]":
                flush_para()
                j = i + 1
                rows = []
                while j < len(lines) and lines[j].strip() != "[/TABLE]":
                    if lines[j].strip():
                        rows.append(lines[j].strip())
                    j += 1
                if rows:
                    blocks.append(Block("table", "\n".join(rows), page))
                i = j + 1
                continue
            if ln.startswith("## "):
                flush_para()
                blocks.append(Block("heading", ln[3:].strip(), page))
            elif not ln:
                flush_para()
            elif looks_like_heading(ln) and not para:
                blocks.append(Block("heading", ln, page))
            else:
                para.append(ln)
            i += 1
        flush_para()
    seen: set[str] = set()
    unique = []
    for b in blocks:
        key = " ".join(b.text.lower().split())
        if b.kind != "heading" and (key in seen or key.startswith("lorem ipsum")):
            continue
        seen.add(key)
        unique.append(b)
    # a heading followed directly by another heading (its section was a duplicate) is dropped too
    return [b for i, b in enumerate(unique)
            if not (b.kind == "heading" and (i + 1 == len(unique) or unique[i + 1].kind == "heading"))]


def split_long(text: str, limit: int = MAX_WORDS) -> list[str]:
    parts, cur = [], []
    for sent in SENTENCE_END.split(text):
        n = len(sent.split())
        if cur and len(" ".join(cur).split()) + n > limit:
            parts.append(" ".join(cur))
            cur = []
        if n > limit:  # one huge "sentence" (no punctuation): cut by words
            words = sent.split()
            for k in range(0, len(words), limit):
                parts.append(" ".join(words[k:k + limit]))
            continue
        cur.append(sent)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_words = 0
    cur_page = 0
    section = ""
    headings: list[str] = []  # headings seen since the last flush

    def flush():
        nonlocal cur, cur_words
        if cur_words:
            chunks.append(Chunk("\n".join(cur), cur_page, section))
        cur, cur_words = [], 0

    def add(text: str, page: int):
        nonlocal cur_words, cur_page
        if not cur_words:
            cur_page = page
            if headings:
                cur.extend(headings)
                cur_words += sum(len(h.split()) for h in headings)
                headings.clear()
            elif section:
                cur.append(section)  # repeat the section title so the passage stands alone
                cur_words += len(section.split())
        cur.append(text)
        cur_words += len(text.split())

    for b in blocks:
        if b.kind == "heading":
            if cur_words >= TAIL_WORDS:
                flush()
            if cur_words:  # short passage so far: keep the heading inside it
                cur.append(b.text)
                cur_words += b.words
            else:
                headings.append(b.text)
            section = b.text
            continue
        if b.kind == "table":
            if cur_words and cur_words + b.words > MAX_WORDS:
                flush()
            add(b.text, b.page)
            if cur_words >= MIN_WORDS:
                flush()
            continue
        pieces = split_long(b.text) if b.words > MAX_WORDS else [b.text]
        for piece in pieces:
            n = len(piece.split())
            if cur_words and cur_words + n > MAX_WORDS:
                flush()
            add(piece, b.page)
    flush()
    if len(chunks) >= 2 and chunks[-1].words < TAIL_WORDS and chunks[-2].words + chunks[-1].words <= MAX_WORDS * 1.2:
        last = chunks.pop()
        chunks[-1].text += "\n" + last.text
    return [c for c in chunks if c.text.strip()]


def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    return chunk_blocks(to_blocks(pages))
