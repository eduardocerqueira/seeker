#date: 2024-09-20T16:38:20Z
#url: https://api.github.com/gists/246d7928b2fc26821db582be583d8b7a
#owner: https://api.github.com/users/tkellogg

import re

# Remove leading and trailing text, leaving just the JSON
_JSON_TRIM_PATTERN = re.compile(
    r"^"                # Start of string
    r"[^{\[]*"          # Drop everything up to { or [
    r"([{\[].*[}\]])"  # Keep the JSON
    # Greedy match here should force it to not consume JSON
    r"[^}\]]*"          # Drop everything after } or ]
    r"$",               # End of string
    re.DOTALL,
)

# Remove invalid escape sequences
# This is bc mixtral seems to be an idiot, and why prompt better when you can just
# fix it.
_JSON_INVALID_ESCAPE_PATTERN = re.compile(
    r"\\([^bfrntu\\/\"])"
)

def make_json_safer():
    """
    Mock out current JSON prep/parsing functionality. Call this once at startup (e.g. after imports)
    """
    orig_fn = dspy.functional._unwrap_json
    def _safer_unwrap_json(output, from_json: Callable[[str], Union[pydantic.BaseModel, str]]):
        output = _JSON_TRIM_PATTERN.sub("\\1", output)
        output = _JSON_INVALID_ESCAPE_PATTERN.sub("\\1", output)
        try:
            return orig_fn(output, from_json)
        except:
            logger.debug(output)
            raise

    if dspy.functional._unwrap_json != _safer_unwrap_json:
        dspy.functional._unwrap_json = _safer_unwrap_json