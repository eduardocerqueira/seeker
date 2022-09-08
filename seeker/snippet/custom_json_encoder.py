#date: 2022-09-08T17:15:13Z
#url: https://api.github.com/gists/93d5bf8c56e3b2935c62a8603cc06486
#owner: https://api.github.com/users/xultaeculcis

# -*- coding: utf-8 -*-
from datetime import date, datetime
from json import JSONEncoder
from pathlib import Path
from typing import Any


class JsonEncoder(JSONEncoder):
    """Custom JSON encoder that handles datatypes that are not out-of-the-box supported by the `json` package."""

    def default(self, o: Any) -> str:
        if isinstance(o, datetime) or isinstance(o, date):
            return o.isoformat()

        if isinstance(o, Path):
            return o.as_posix()

        return super().default(o)
