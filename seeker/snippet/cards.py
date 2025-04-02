#date: 2025-04-02T17:02:52Z
#url: https://api.github.com/gists/ef8989a7050f0a89a68405516c46f9e0
#owner: https://api.github.com/users/codeshard

import re

(
    CUENTA_MLC_METRO,
    TARJETA_MLC_METRO,
    CUENTA_MLC_BPA,
    TARJETA_MLC_BPA,
    CUENTA_MLC_BANDEC,
    TARJETA_MLC_BANDEC,
) = (
    "Cuenta MLC Banco Metropolitano",
    "Tarjeta MLC Banco Metropolitano",
    "Cuenta MLC Banco Popular de Ahorro",
    "Tarjeta MLC Banco Polular de Ahorro",
    "Cuenta MLC Banco de Crédico y Comercio",
    "Tarjeta MLC Banco de Crédico y Comercio",
)

CARD_TYPES = [
    (CUENTA_MLC_METRO, (19,), (r"^05\d{2}-7\d{3}-\d{4}-\d{4}$")),
    (TARJETA_MLC_BPA, (19,), (r"^12\d{2}-\d{4}-\d{4}-\d{4}$")),
    (CUENTA_MLC_BANDEC, (19,), (r"^06\d{2}-\d{4}-\d{4}-\d{4}$")),
    (TARJETA_MLC_METRO, (19,), (r"^92\d{2}-9598-7\d{3}-\d{4}$")),
    (CUENTA_MLC_BPA, (19,), (r"^92\d{2}-1299-7\d{3}-\d{4}$")),
    (TARJETA_MLC_BANDEC, (19,), (r"^92\d{2}-0699-9\d{3}-\d{4}$")),
]


def bankcard_type(card_number: str) -> str:
    def matches(card_number, lengths, card_regex) -> bool:
        if len(card_number) not in lengths:
            return False
        if re.match(card_regex, card_number):
            return True
        return False

    for card_type, lengths, card_regex in CARD_TYPES:
        if matches(card_number, lengths, card_regex):
            return card_type
