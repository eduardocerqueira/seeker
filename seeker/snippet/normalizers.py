#date: 2026-03-03T17:28:18Z
#url: https://api.github.com/gists/a9c87a8170a502a450c3c1cdf94df6bd
#owner: https://api.github.com/users/ipritamdash

"""
normalizers.py — Text normalization for CER/WER evaluation.

Greek-specific normalizer + generic fallback for other languages.
Uses jiwer (pip install jiwer==3.0.3) for actual CER/WER computation.

Usage:
    from normalizers import normalize_greek, normalize_generic, compute_cer_wer

    refs = [normalize_greek(r) for r in references]
    preds = [normalize_greek(p) for p in predictions]
    scores = compute_cer_wer(refs, preds)
"""

import re
import unicodedata


def normalize_greek(text: str) -> str:
    """
    8-step Greek normalizer. Handles diacritics, final sigma,
    homoglyphs, and apostrophe variants common in Greek text.
    """
    # NFKC — resolve compatibility chars (ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Apostrophe variants → ASCII (U+2019, U+2018, U+02BC, U+02BB)
    for ch in "\u2019\u2018\u02bc\u02bb":
        text = text.replace(ch, "'")

    text = text.lower()

    # Strip combining diacriticals (accents, breathing marks)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text)

    # Final sigma ς → σ
    text = text.replace("\u03c2", "\u03c3")

    # Greek ο (U+03BF) → Latin o (only truly confusable lowercase pair)
    text = text.translate(str.maketrans({"\u03bf": "o"}))

    # Strip all punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    return re.sub(r"\s+", " ", text).strip()


def normalize_generic(text: str) -> str:
    """
    Minimal normalizer for non-Greek languages.
    Lowercase, strip punctuation, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def compute_cer_wer(references: list[str], predictions: list[str]) -> dict:
    """
    Corpus-level + per-sample CER/WER using jiwer.
    No capping — hallucination rows can exceed 100%. Flag them separately.
    """
    from jiwer import wer, cer

    corpus_wer = wer(references, predictions)
    corpus_cer = cer(references, predictions)

    per_sample = []
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        s_wer = wer(ref, pred)
        s_cer = cer(ref, pred)
        per_sample.append({
            "index": i,
            "wer": round(s_wer, 4),
            "cer": round(s_cer, 4),
            "hallucination": s_cer > 1.0,  # flag run-on / hallucinated outputs
        })

    return {
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "per_sample": per_sample,
        "n_hallucinations": sum(1 for s in per_sample if s["hallucination"]),
    }


if __name__ == "__main__":
    # quick sanity check
    refs = ["ο κοσμοσ ειναι ωραιοσ", "hello world"]
    preds = ["ο κοσμοσ ειναι ωραιοσ", "helo wrld"]

    norm_refs = [normalize_greek(r) for r in refs]
    norm_preds = [normalize_greek(p) for p in preds]

    print("Normalized refs:", norm_refs)
    print("Normalized preds:", norm_preds)
    print("Scores:", compute_cer_wer(norm_refs, norm_preds))
