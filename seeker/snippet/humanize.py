#date: 2025-12-10T16:56:05Z
#url: https://api.github.com/gists/7c1cf4083791658358885844b40ed907
#owner: https://api.github.com/users/BJWOODS

# credit: https://github.com/Nordth/humanize-ai-lib/blob/main/src/humanize-string.ts
import re

_HIDDEN_CHARS = re.compile(
    r"[\u00AD\u180E\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]"
)
_TRAILING_WS  = re.compile(r"[ \t\x0B\f]+$", re.MULTILINE)
_NBSP         = re.compile(r"\u00A0")
_DASHES       = re.compile(r"[—–]+")         # em- & en-dashes → ASCII hyphen
_DQUOTES      = re.compile(r"[“”«»„]")       # curly / guillemets → "
_SQUOTES      = re.compile(r"[‘’ʼ]")         # curly apostrophes → '
_ELLIPSIS     = re.compile(r"…")             # single‐char ellipsis → "..."

def humanize_str(text: str) -> str:
    text = _HIDDEN_CHARS.sub("", text)
    text = _TRAILING_WS.sub("", text)
    text = _NBSP.sub(" ", text)
    text = _DASHES.sub("-", text)
    text = _DQUOTES.sub('"', text)
    text = _SQUOTES.sub("'", text)
    text = _ELLIPSIS.sub("...", text)
    return text
