#date: 2026-01-16T17:11:25Z
#url: https://api.github.com/gists/6507233512aa22a8a3dfed0daf6f4955
#owner: https://api.github.com/users/rj9889

import json
import re
from typing import Dict, List, Any


# -----------------------------
# Text normalization helpers
# -----------------------------

WS_RE = re.compile(r"[ \t\u00A0]+")
MANY_NL_RE = re.compile(r"\n{3,}")
HYPHEN_NL_RE = re.compile(r"(\w)-\n(\w)")  # exam-\nple -> example


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # fix hyphenated line breaks
    text = HYPHEN_NL_RE.sub(r"\1\2", text)

    lines = []
    for line in text.split("\n"):
        line = WS_RE.sub(" ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = MANY_NL_RE.sub("\n\n", text)

    return text.strip()


# -----------------------------
# OCR extraction
# -----------------------------

def extract_lines(doc: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Returns {page_number: [lines]}
    Supports Azure, Read API, or generic OCR JSON
    """
    pages = {}

    analyze = doc.get("analyzeResult", doc)

    # Azure Document Intelligence
    if "pages" in analyze:
        for p in analyze["pages"]:
            page = p.get("pageNumber", 1)
            pages.setdefault(page, [])
            for line in p.get("lines", []):
                text = line.get("content") or line.get("text")
                if text:
                    pages[page].append(text)

    # Read API style
    elif "readResults" in analyze:
        for p in analyze["readResults"]:
            page = p.get("page", 1)
            pages.setdefault(page, [])
            for line in p.get("lines", []):
                text = line.get("text")
                if text:
                    pages[page].append(text)

    # Fallback: raw content
    else:
        content = analyze.get("content") or doc.get("content", "")
        pages[1] = [l for l in content.split("\n") if l.strip()]

    return pages


# -----------------------------
# Header / footer removal
# -----------------------------

def remove_headers_footers(pages: Dict[int, List[str]], min_repeat: int = 3):
    if len(pages) < min_repeat:
        return pages

    from collections import Counter

    def norm(s: str):
        s = s.lower()
        s = re.sub(r"\d+", "#", s)
        return WS_RE.sub(" ", s).strip()

    headers = Counter()
    footers = Counter()

    for lines in pages.values():
        if lines:
            headers[norm(lines[0])] += 1
            footers[norm(lines[-1])] += 1

    bad_headers = {k for k, v in headers.items() if v >= min_repeat}
    bad_footers = {k for k, v in footers.items() if v >= min_repeat}

    cleaned = {}
    for page, lines in pages.items():
        new_lines = []
        for i, line in enumerate(lines):
            key = norm(line)
            if i == 0 and key in bad_headers:
                continue
            if i == len(lines) - 1 and key in bad_footers:
                continue
            new_lines.append(line)
        cleaned[page] = new_lines

    return cleaned


# -----------------------------
# Build final text
# -----------------------------

def build_text(pages: Dict[int, List[str]]) -> str:
    out = []
    for page in sorted(pages):
        out.append(f"--- PAGE {page} ---")
        out.extend(pages[page])
        out.append("")
    return normalize_text("\n".join(out))


# -----------------------------
# Main function
# -----------------------------

def ocr_json_to_txt(
    input_json: str,
    output_txt: str = "normalized.txt",
    remove_hf: bool = True
):
    with open(input_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    pages = extract_lines(doc)

    if remove_hf:
        pages = remove_headers_footers(pages)

    final_text = build_text(pages)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(final_text + "\n")

    print(f"✅ Normalized text written to: {output_txt}")


# -----------------------------
# CLI usage
# -----------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_to_txt_normalizer.py <ocr_json>")
        exit(1)

    ocr_json_to_txt(sys.argv[1])
