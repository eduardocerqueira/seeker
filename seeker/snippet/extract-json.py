#date: 2025-01-23T16:53:56Z
#url: https://api.github.com/gists/2eaa5bc5d773e07966f5e24d08f654cc
#owner: https://api.github.com/users/arose13

import re

def extract_json(text):
    """
    Extracts the first JSON object from a given string.
    
    Args:
        text (str): The input string containing JSON.
    
    Returns:
        str: The extracted JSON as a string, or None if no JSON is found.
    """
    regex = r"\{(?:[^{}]|\"(?:\\.|[^\"\\])*\"|(?R))*\}"
    match = re.search(regex, text)
    return match.group(0) if match else None