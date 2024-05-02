#date: 2024-05-02T16:59:31Z
#url: https://api.github.com/gists/4be889c448a26b7c4df669af4d496496
#owner: https://api.github.com/users/tyschacht

import json
import re


def parse_json_from_gemini(json_str: str):
    """Parses a dictionary from a JSON-like object string.

    Args:
      json_str: A string representing a JSON-like object, e.g.:
        ```json
        {
          "key1": "value1",
          "key2": "value2"
        }
        ```

    Returns:
      A dictionary representing the parsed object, or None if parsing fails.
    """

    try:
        # Remove potential leading/trailing whitespace
        json_str = json_str.strip()

        # Extract JSON content from triple backticks and "json" language specifier
        json_match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)

        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        return None
