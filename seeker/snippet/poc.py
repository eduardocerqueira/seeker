#date: 2025-06-16T17:09:01Z
#url: https://api.github.com/gists/6e9738c3f3fe45a687cebe48347162b2
#owner: https://api.github.com/users/Rye-Catcher

import hashlib
from decimal import Decimal
import json
from pathlib import Path
from typing import List, Dict

EMPTY_FIELDS = ("Strike", "IssuePrice")          # the “solve” columns
PRECISION     = Decimal("0.01")                # 2-dp canonical format

def _canonical(v):
    """
    Normalise a single value so round-trips through e-mail/CSV
    won't change the bytes that go into the hash.
    """
    if v is None:
        return ""
    if isinstance(v, (float, Decimal)):
        return str(Decimal(str(v)).quantize(PRECISION))
    return str(v).strip().lower()                # trim & case-fold strings

def make_signature(request: dict) -> str:
    """
    Return either:
        STRIKE@<sha256>    if Strike is empty and IssuePrice is filled
        ISSUEPRICE@<sha256>  vice-versa
    Raises if both / neither are empty.
    """
    empties = [fld for fld in EMPTY_FIELDS if not request.get(fld)]
    if len(empties) != 1:
        raise ValueError("Exactly one of Strike or IssuePrice must be empty")

    empty_field = empties[0]                     # “solve” column
    # keep all other key-value pairs, canonicalise & sort deterministically
    payload = "|".join(
        f"{k.lower()}={_canonical(request[k])}"
        for k in sorted(request) if k != empty_field
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{empty_field.upper()}@{digest}"

def _digest(row: Dict, drop_key: str) -> str:
    """
    Hash of all key=value pairs except *drop_key*.
    Stable order: alphabetical keys, lower-cased.
    """
    payload = "|".join(
        f"{k.lower()}={_canonical(row[k])}"
        for k in sorted(row) if k.lower() != drop_key
    )
    return hashlib.sha256(payload.encode()).hexdigest()

def find_matches(
        response_rows: List[Dict],
        signature: str,
) -> List[Dict]:
    """
    Return a list of provider rows that match the outbound *signature*.
    Signature format examples:  STRIKE@<hex-digest>   or   ISSUEPRICE@<digest>
    """
    try:
        drop_key, req_hash = signature.split("@", 1)
    except ValueError:
        raise ValueError("Signature must look like  STRIKE@<hash>  or  ISSUEPRICE@<hash>")

    drop_key = drop_key.strip().lower()  # normalise
    matches = []

    for row in response_rows:
        row_hash = _digest(row, drop_key)
        print(f'res# {row_hash}')
        if row_hash == req_hash:
            matches.append(row)

    return matches

sample_sending_request = {
    "DaysToIssue": "T+7",
    "IssueDate": "24/06/2025",
    "CCY": "USD",
    "MaturityPeriod": 3,
    "FixingDate": "24/09/25",
    "SettlementDate": "26/09/25",
    "Underlyings": "AAPL US+MSFT US",
    "Strike": 98,
    "IssuePrice": "",
    "NotionalAmount": 500000.00,
}

response_path = Path("response.json")            # adjust if needed
with response_path.open(encoding="utf-8") as f:
    responses = json.load(f)

sig = make_signature(sample_sending_request)
print(sig)


hits = find_matches(responses, sig)

if hits:
    print(f"\n{len(hits)} matching row(s) found:\n")
    for ix, row in enumerate(hits, 1):
        print(f"─ Row #{ix} ─")
        for k, v in row.items():
            print(f"{k:<15}: {v}")
        print()
else:
    print("No match found.")
