#date: 2023-09-18T17:05:08Z
#url: https://api.github.com/gists/0fdc12fcd3f3649b04908cc32f3173bb
#owner: https://api.github.com/users/clemlesne

from typing import Optional
import re
import html


def sanitize(raw: Optional[str]) -> Optional[str]:
    """
    Takes a raw string of HTML and removes all HTML tags, Markdown tables, and line returns.
    """
    if not raw:
        return None

    # Remove HTML doctype
    raw = re.sub(r"<!DOCTYPE[^>]*>", " ", raw)
    # Remove HTML head
    raw = re.sub(r"<head\b[^>]*>[\s\S]*<\/head>", " ", raw)
    # Remove HTML scripts
    raw = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", " ", raw)
    # Remove HTML styles
    raw = re.sub(r"<style\b[^>]*>[\s\S]*?<\/style>", " ", raw)
    # Extract href from HTML links, in the form of "(href) text"
    raw = re.sub(r"<a\b[^>]*href=\"([^\"]*)\"[^>]*>([^<]*)<\/a>", r"(\1) \2", raw)
    # Remove HTML tags
    raw = re.sub(r"<[^>]*>", " ", raw)
    # Remove Markdown tables
    raw = re.sub(r"[-|]{2,}", " ", raw)
    # Remove Markdown code blocks
    raw = re.sub(r"```[\s\S]*```", " ", raw)
    # Remove Markdown bold, italic, strikethrough, code, heading, table delimiters, links, images, comments, and horizontal rules
    raw = re.sub(r"[*_`~#|!\[\]<>-]+", " ", raw)
    # Remove line returns, tabs and spaces
    raw = re.sub(r"[\n\t\v ]+", " ", raw)
    # Remove HTML entities
    raw = html.unescape(raw)
    # Remove leading and trailing spaces
    raw = raw.strip()

    return raw
