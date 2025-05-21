#date: 2025-05-21T16:43:45Z
#url: https://api.github.com/gists/67755bd650af80ee2ee630c0f0b45329
#owner: https://api.github.com/users/rbpaul2

import re

def convert_lyrics_format(lyrics: str) -> str:
    # Match lines like "🎸 Chorus:" or "🎤 Verse:" and replace with [Chorus], [Verse], etc.
    def replace_section_header(match):
        section_name = match.group(1).strip()
        return f'[{section_name}]'

    # Use regex to identify any line starting with an emoji and a section name followed by a colon
    pattern = re.compile(r'^\s*[\U0001F300-\U0001FAFF]?\s*(\w+):', re.MULTILINE)
    
    return pattern.sub(replace_section_header, lyrics)
