#date: 2025-09-04T16:48:20Z
#url: https://api.github.com/gists/2d3620d39e7d9fb55157a6e3471606f3
#owner: https://api.github.com/users/docsallover

import re

text = "This is a test. My phone number is 123-456-7890. This is another test."
pattern = r"test"
replacement = "sample"

# 1. re.search() - Finds the first match anywhere in the string
search_match = re.search(pattern, text)
print(f"re.search(): {search_match.group() if search_match else None}")

# 2. re.match() - Tries to match only at the beginning of the string
match_match = re.match(pattern, text)
print(f"re.match(): {match_match.group() if match_match else None}")

# 3. re.findall() - Finds all occurrences and returns a list
all_matches = re.findall(pattern, text)
print(f"re.findall(): {all_matches}")

# 4. re.sub() - Finds and replaces all matches
substituted_text = re.sub(pattern, replacement, text)
print(f"re.sub(): {substituted_text}")

# 5. re.split() - Splits the string by the pattern
split_list = re.split(pattern, text)
print(f"re.split(): {split_list}")