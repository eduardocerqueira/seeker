#date: 2026-01-16T17:09:23Z
#url: https://api.github.com/gists/008f023e3111fab569ee238d7db0d051
#owner: https://api.github.com/users/rj9889

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Helpers: text cleaning
# -----------------------------

_WS_RE = re.compile(r"[ \t\u00A0]+")
_MANY_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")  # word-\nword -> wordword


def normalize_text_block(text: str) -> str:
    """
    Normalize OCR text while preserving paragraph structure.
    """
    if not text:
        return ""

    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces on lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # De-hyphenate common line-wrap hyphenations: exam-\nple -> example
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Collapse weird whitespace within lines
    lines = []
    for line in text.split("\n"):
        line = _WS_RE.sub(" ", line).strip()
        lines.append(line)

    text = "\n".join(lines)

    # Collapse too many blank lines
    text = _MANY_NL_RE.sub("\n\n", text).strip()

    return text


# -----------------------------
# OCR JSON parsing (robust)
# -----------------------------

@dataclass
class LineItem:
    page: int
    text: str
    confidence: Optional[float] = None


def extract_lines_from_ocr_json(doc: Dict[str, Any]) -> List[LineItem]:
    """
    Extract lines in reading order.
    Prefers page/line structures if available; falls back to doc['content'] splitlines.
    Handles common Azure Document Intelligence formats.
    """
    lines: List[LineItem] = []

    # Common Azure DI structure:
    # doc["analyzeResult"]["pages"][i]["lines"][j]["content"]
    analyze = doc.get("analyzeResult") or doc  # allow passing analyzeResult directly

    pages = analyze.get("pages")
    if isinstance(pages, list) and pages:
        for p in pages:
            page_num = int(p.get("pageNumber") or p.get("page") or 1)
            p_lines = p.get("lines") or []
            if isinstance(p_lines, list) and p_lines:
                for ln in p_lines:
                    txt = ln.get("content") or ln.get("text") or ""
                    conf = ln.get("confidence")
                    if txt:
                        lines.append(LineItem(page=page_num, text=txt, confidence=conf))

    # Another common format: doc["readResults"][i]["lines"][j]["text"]
    if not lines:
        read_results = analyze.get("readResults")
        if isinstance(read_results, list) and read_results:
            for p in read_results:
                page_num = int(p.get("page") or p.get("pageNumber") or 1)
                p_lines = p.get("lines") or []
                for ln in p_lines:
                    txt = ln.get("text") or ln.get("content") or ""
                    conf = ln.get("confidence")
                    if txt:
                        lines.append(LineItem(page=page_num, text=txt, confidence=conf))

    # Fallback: plain content field with \n
    if not lines:
        content = analyze.get("content") or doc.get("content") or ""
        if content:
            for i, t in enumerate(content.splitlines(), start=1):
                # If you don't have page mapping, treat as page 1
                if t.strip():
                    lines.append(LineItem(page=1, text=t.strip(), confidence=None))

    # Sort by page (reading order). Within a page, OCR usually already ordered.
    lines.sort(key=lambda x: (x.page,))
    return lines


# -----------------------------
# Header/footer removal
# -----------------------------

def remove_repeating_headers_footers(
    lines: List[LineItem],
    header_top_n: int = 2,
    footer_bottom_n: int = 2,
    min_repeat_pages: int = 3
) -> List[LineItem]:
    """
    Detect repeating header/footer lines across pages and remove them.
    Works best when OCR gives page numbers.
    """
    if not lines:
        return lines

    # group by page
    by_page: Dict[int, List[str]] = {}
    for li in lines:
        by_page.setdefault(li.page, []).append(li.text)

    pages = sorted(by_page.keys())
    if len(pages) < min_repeat_pages:
        return lines

    # collect candidate header/footer lines
    header_candidates: List[str] = []
    footer_candidates: List[str] = []

    for p in pages:
        plines = by_page[p]
        header_candidates.extend(plines[:header_top_n])
        footer_candidates.extend(plines[-footer_bottom_n:])

    def normalize_key(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\d+", "#", s)  # mask digits (page numbers/dates)
        s = _WS_RE.sub(" ", s)
        return s

    # count occurrences
    from collections import Counter
    hc = Counter(normalize_key(x) for x in header_candidates if x.strip())
    fc = Counter(normalize_key(x) for x in footer_candidates if x.strip())

    # treat as repeating if appears on many pages
    header_keys = {k for k, c in hc.items() if c >= min_repeat_pages}
    footer_keys = {k for k, c in fc.items() if c >= min_repeat_pages}

    filtered: List[LineItem] = []
    for li in lines:
        k = normalize_key(li.text)
        if k in header_keys or k in footer_keys:
            continue
        filtered.append(li)

    return filtered


# -----------------------------
# Build normalized document text
# -----------------------------

def build_document_text(lines: List[LineItem]) -> str:
    """
    Join lines into a document with page breaks.
    """
    if not lines:
        return ""

    out: List[str] = []
    current_page = lines[0].page
    out.append(f"--- PAGE {current_page} ---")

    for li in lines:
        if li.page != current_page:
            current_page = li.page
            out.append("")  # blank line between pages
            out.append(f"--- PAGE {current_page} ---")
        out.append(li.text)

    return normalize_text_block("\n".join(out))


# -----------------------------
# Chunking for vector/KBS
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = 1800,   # chars
    overlap: int = 250        # chars
) -> List[str]:
    """
    Fast, stable chunking by characters with overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)

        # try to end on a boundary for nicer chunks
        boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
        if boundary > start + int(chunk_size * 0.6):
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        start = max(0, end - overlap)

    return chunks


def write_chunks_jsonl(
    chunks: List[str],
    out_path: str,
    doc_id: str,
    extra_meta: Optional[Dict[str, Any]] = None
) -> None:
    extra_meta = extra_meta or {}
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            rec = {
                "id": f"{doc_id}::chunk::{i:05d}",
                "doc_id": doc_id,
                "chunk_index": i,
                "text": ch,
                **extra_meta,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -----------------------------
# Main entry
# -----------------------------

def normalize_ocr_json_to_kbs_files(
    ocr_json_path: str,
    out_txt_path: str = "normalized.txt",
    out_chunks_path: str = "chunks.jsonl",
    doc_id: str = "doc",
    remove_headers_footers: bool = True,
) -> Tuple[str, int]:
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    lines = extract_lines_from_ocr_json(doc)

    if remove_headers_footers:
        lines = remove_repeating_headers_footers(lines)

    full_text = build_document_text(lines)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text + "\n")

    chunks = chunk_text(full_text, chunk_size=1800, overlap=250)
    write_chunks_jsonl(chunks, out_chunks_path, doc_id=doc_id)

    return out_txt_path, len(chunks)


if __name__ == "__main__":
    # Example usage:
    # python normalize_ocr.py input.json
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "ocr.json"
    txt_path, n_chunks = normalize_ocr_json_to_kbs_files(
        ocr_json_path=inp,
        out_txt_path="normalized.txt",
        out_chunks_path="chunks.jsonl",
        doc_id="my_document",
        remove_headers_footers=True,
    )
    print(f"Saved: {txt_path} | chunks: {n_chunks} -> chunks.jsonl")
