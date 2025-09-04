#date: 2025-09-04T16:45:18Z
#url: https://api.github.com/gists/6565f5feaafee065219ecf4e33cae4ba
#owner: https://api.github.com/users/docsallover

import re

text = "Invoice for 2025-09-04 was paid. Next invoice due 2026-01-15."
# \d matches any digit (0-9)
# {4} matches exactly 4 digits
# The parentheses create capturing groups for year, month, and day

pattern = r"(\d{4})-(\d{2})-(\d{2})"

# Find all matches
all_matches = re.findall(pattern, text)
print(f"All captured dates: {all_matches}")

# Find and capture the first match
search_match = re.search(pattern, text)
if search_match:
    print(f"\nFirst full match: {search_match.group(0)}")
    print(f"Captured year: {search_match.group(1)}")
    print(f"Captured month: {search_match.group(2)}")
    print(f"Captured day: {search_match.group(3)}")