#date: 2021-11-29T17:07:26Z
#url: https://api.github.com/gists/cc65872bb1f3ef6d6009dd56fcd060a9
#owner: https://api.github.com/users/davidverweij

from functools import reduce
import re
from typing import Any, Optional

def find_key(dot_notation_path: str, payload: dict) -> Any:
    """Try to get a deep value from a dict based on a dot-notation"""

    def get_despite_none(payload: Optional[dict], key: str) -> Any:
        """Try to get value from dict, even if dict is None"""
        if not payload or not isinstance(payload, (dict, list)):
            return None
        # can also access lists if needed, e.g., if key is '[1]'
        if (num_key := re.match(r"^\[(\d+)\]$", key)) is not None:
            try:
                return payload[int(num_key.group(1))]
            except IndexError:
                return None
        else:
            return payload.get(key, None)

    found = reduce(get_despite_none, dot_notation_path.split("."), payload)
   
    # compare to None, as the key could exist and be empty
    if found is None:
        raise KeyError()
    return found
    
"""In my use case, I need to find a key within an HTTP request payload, which can often include lists as well. The following examples work:"""

payload = {
    "haystack1": {
        "haystack2": {
            "haystack3": None, 
            "haystack4": "needle"
        }
    },
    "haystack5": [
        {"haystack6": None}, 
        {"haystack7": "needle"}
    ],
    "haystack8": {},
}

find_key("haystack1.haystack2.haystack4", payload)
# "needle"
find_key("haystack5.[1].haystack7", payload)
# "needle"
find_key("[0].haystack5.[1].haystack7", [payload, None])
# "needle"
find_key("haystack8", payload)
# {}
find_key("haystack1.haystack2.haystack4.haystack99", payload)
# KeyError