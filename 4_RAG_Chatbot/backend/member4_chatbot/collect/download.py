"""Download the knowledge-base documents listed in sources.yaml into data/docs/raw/.

    python -m member4_chatbot.collect.download          # fetch what is missing
    python -m member4_chatbot.collect.download --force  # fetch everything again

- doa.gov.lk pages are saved twice: the raw `.html` and a `.txt` of the page's main content, with each table row
  written as `cell | cell | cell` so the tables survive as text.
- Google Drive PDFs are saved as `.pdf` (or `.jpg`/`.png` if the DOA shared an image).
- `manifest.yaml` gets one entry per document: title, URL, language, publisher, date retrieved, file, status.
  A failed download stays in the manifest with `status: failed` and the error, so the user can fetch it by hand
  and drop the file into data/docs/raw/ under the listed name.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

from member4_chatbot.paths import RAW_DOCS_DIR, SOURCES_YAML

UA = {"User-Agent": "Mozilla/5.0 (SoilMind university project; knowledge-base collection)"}
DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
DRIVE_VIEW = "https://drive.google.com/file/d/{id}/view"
MAGIC = {b"%PDF": ".pdf", b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png"}


def html_to_text(html: str) -> str:
    """Main content of a doa.gov.lk (WordPress / Elementor) page as plain text; tables become `a | b | c` rows."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.find(attrs={"data-elementor-type": "wp-page"}) or soup.body
    for tag in main.find_all(["script", "style", "noscript", "nav", "header", "footer", "form"]):
        tag.decompose()
    for table in main.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n"))
    for h in main.find_all(re.compile(r"^h[1-6]$")):
        h.replace_with(soup.new_string(f"\n## {h.get_text(' ', strip=True)}\n"))
    text = main.get_text("\n")
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return "\n".join(out).strip() + "\n"


def save_html(html: str, dest: Path) -> Path:
    txt = html_to_text(html)
    if len(txt) < 200:
        raise ValueError(f"page has almost no text ({len(txt)} chars)")
    dest.with_suffix(".html").write_text(html, encoding="utf-8")
    dest.with_suffix(".txt").write_text(txt, encoding="utf-8")
    return dest.with_suffix(".txt")


def fetch_html(url: str, dest: Path) -> tuple[Path, int]:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    return save_html(r.text, dest), len(r.content)


def fetch_drive(drive_id: str, dest: Path) -> tuple[Path, int]:
    r = requests.get(DRIVE_URL.format(id=drive_id), headers=UA, timeout=300)
    r.raise_for_status()
    body = r.content
    ext = next((e for m, e in MAGIC.items() if body.startswith(m)), None)
    if ext is None:
        snippet = body[:200].decode("utf-8", "replace").replace("\n", " ")
        raise ValueError(f"not a PDF or image (Drive returned: {snippet[:120]!r})")
    path = dest.with_suffix(ext)
    path.write_bytes(body)
    return path, len(body)


SITE_HOSTS = ("doa.gov.lk", "agridpt.w3dtec.net", "..")


def page_slug(href: str) -> str | None:
    """`http://agridpt.w3dtec.net/rrdi_pests_thrips/` or `../rrdi_pests_thrips` -> `rrdi_pests_thrips`."""
    m = re.match(r"^(?:https?://([^/]+))?/?(?:\.\./)?(?:wp-content/)?([A-Za-z0-9_\-]+)/?(?:[?#].*)?$", href.strip())
    if not m or (m.group(1) and m.group(1) not in SITE_HOSTS and not m.group(1).endswith("doa.gov.lk")):
        return None
    return m.group(2)


def crawl(spec: dict) -> list[dict]:
    """Breadth-first crawl of doa.gov.lk pages whose slug matches `allow`; returns item dicts like sources.yaml's."""
    allow, deny = re.compile(spec["allow"]), re.compile(spec["deny"])
    queue, seen, items = list(spec["seeds"]), set(spec["seeds"]), []
    while queue and len(items) < spec["max_pages"]:
        slug = queue.pop(0)
        url = f"https://doa.gov.lk/{slug}/"
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.encoding = "utf-8"
        except requests.RequestException as e:
            items.append({"id": slug.replace("-", "_"), "title": slug, "lang": "unknown", "url": url, "error": str(e)})
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.get_text() if soup.title else slug).split(" – ")[0].strip()
        if r.status_code != 200 or "not found" in title.lower():
            continue
        main = soup.find("main") or soup.body
        for a in main.find_all("a"):
            s = page_slug(a.get("href", ""))
            if s and s not in seen and allow.search(s) and not deny.search(s):
                seen.add(s)
                queue.append(s)
        if not deny.search(slug):
            lang = "si" if re.search(r"_si(-\d+)?$|_si_", slug) else "en"
            items.append({"id": slug.replace("-", "_"), "title": title, "lang": lang, "url": url, "html": r.text})
        time.sleep(0.3)
    return items


def existing_file(dest: Path) -> Path | None:
    for ext in (".txt", ".pdf", ".jpg", ".png"):
        if dest.with_suffix(ext).exists():
            return dest.with_suffix(ext)
    return None


def main(force: bool = False) -> dict:
    sources = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8"))
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DOCS_DIR / "manifest.yaml"
    old = {}
    if manifest_path.exists():
        old = {d["id"]: d for d in (yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}).get("documents", [])}
    today = dt.date.today().isoformat()
    docs = []
    for group, spec in sources.items():
        if spec["kind"] == "crawl":
            crawled = [d for d in old.values() if d.get("group") == group and d.get("status") == "ok"]
            if crawled and not force:
                items = [{k: d[k] for k in ("id", "title", "url")} | {"lang": d["language"]} for d in crawled]
            else:
                items = crawl(spec)
                print(f"crawl   {group}: {len(items)} pages")
        else:
            items = spec["items"]
        for item in items:
            dest = RAW_DOCS_DIR / item["id"]
            url = item.get("url") or DRIVE_VIEW.format(id=item["drive_id"])
            entry = {
                "id": item["id"], "title": item["title"], "url": url, "listed_on": spec["listed_on"],
                "language": item["lang"], "publisher": spec["publisher"], "group": group, "kind": spec["kind"],
            }
            have = existing_file(dest)
            if have and not force:
                prev = old.get(item["id"], {})
                entry.update(file=have.name, bytes=have.stat().st_size, status="ok",
                             date_retrieved=prev.get("date_retrieved", today))
                if prev.get("status") == "manual":
                    entry["status"] = "manual"
                docs.append(entry)
                continue
            try:
                if "error" in item:
                    raise RuntimeError(item["error"])
                if "html" in item:  # already fetched by the crawl
                    path, n = save_html(item["html"], dest), len(item["html"].encode("utf-8"))
                elif spec["kind"] in ("html", "crawl"):
                    path, n = fetch_html(url, dest)
                else:
                    path, n = fetch_drive(item["drive_id"], dest)
                entry.update(file=path.name, bytes=n, status="ok", date_retrieved=today)
                print(f"ok      {item['id']:<45} {n/1024:8.0f} kB")
            except Exception as e:  # noqa: BLE001 - every failure is recorded in the manifest
                entry.update(file=None, status="failed", error=str(e)[:300], date_retrieved=None)
                print(f"FAILED  {item['id']:<45} {e}")
            docs.append(entry)
            time.sleep(0.5)
    manifest = {
        "description": "Knowledge-base documents for the SoilMind chatbot (Member 4). Built by "
                       "`python -m member4_chatbot.collect.download` from member4_chatbot/collect/sources.yaml.",
        "documents": docs,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=140), encoding="utf-8")
    ok = sum(d["status"] != "failed" for d in docs)
    print(f"{ok}/{len(docs)} documents present; manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    main(ap.parse_args().force)
