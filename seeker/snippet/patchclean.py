#date: 2025-04-14T16:41:11Z
#url: https://api.github.com/gists/9abadff895fbc963b9f6091b99ee4756
#owner: https://api.github.com/users/sciyoshi

#!/usr/bin/env python3
"""
patchclean.py - Process a patch file to handle blank line removals.

This script reads a patch file from stdin, processes it to convert blank line
removals (single "-" lines) to context lines (" "), and updates line numbers
in hunk headers accordingly. It also removes hunks that no longer have changes
and files that no longer have any modification hunks.

Special handling is added for deleted files to preserve their patch structure.

Usage:
    cat patch_file.diff | python patchclean.py > cleaned_patch.diff
"""

import sys
import re


def process_patch(patch_text):
	"""
	Process a patch file to handle blank line removals.

	Args:
		patch_text (str): The content of the patch file

	Returns:
		str: The modified patch with blank line removals converted to context lines
			 and updated line numbers
	"""
	if not patch_text or patch_text.isspace():
		return ""

	lines = patch_text.splitlines()
	result_lines = []

	i = 0
	while i < len(lines):
		# If this is a file header line, keep track of it
		if i < len(lines) and (
			lines[i].startswith("diff ") or lines[i].startswith("--- ") or lines[i].startswith("+++ ")
		):
			file_header_start = i
			while i < len(lines) and not lines[i].startswith("@@ "):
				i += 1

			file_header_lines = lines[file_header_start:i]
			file_hunks = []

			# Check if this is a deleted file by looking for "deleted file" in the header
			is_deleted_file = any("deleted file" in line for line in file_header_lines)

			# Track how many blank line removals we've seen so far
			cumulative_blank_removals = 0

		# If this is a hunk header
		if i < len(lines) and lines[i].startswith("@@ "):
			hunk_header = lines[i]
			i += 1

			# Parse the hunk header to get the line numbers
			match = re.match(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)", hunk_header)
			if match:
				old_start, old_count, new_start, new_count = map(int, match.groups()[:4])
				header_comment = match.group(5)

				# Adjust the starting line numbers based on previous changes
				adjusted_old_start = old_start + cumulative_blank_removals
				adjusted_new_start = new_start

				blank_line_removals = 0
				valid_changes = False
				hunk_body = []

				# Count additions and removals to correctly calculate new line count
				additions = 0
				removals = 0

				# Process the hunk body
				while i < len(lines) and not (lines[i].startswith("@@ ") or lines[i].startswith("diff ")):
					line = lines[i]
					i += 1

					if not is_deleted_file and line.startswith("-") and line.strip() == "-":
						# This is a blank line removal, convert it to a context line
						# Skip this logic for deleted files
						hunk_body.append(" ")
						blank_line_removals += 1
					else:
						hunk_body.append(line)
						# Track additions and removals
						if line.startswith("+"):
							additions += 1
						elif line.startswith("-"):
							removals += 1
						# Check if this line represents an actual change
						if line.startswith("+") or (line.startswith("-") and line.strip() != "-"):
							valid_changes = True

				# Update the header based on changes
				if not is_deleted_file and blank_line_removals > 0:
					# For each blank line removal, we:
					# 1. Reduce the count in the original file (removed the "-" line)
					# 2. Keep the space in the result (blank line is preserved)
					new_old_count = old_count

					# Calculate the correct count for the new file:
					# Start with original count, add blank lines we've preserved,
					# and adjust for any other additions/removals
					adjusted_new_count = new_count + blank_line_removals

					new_header = f"@@ -{adjusted_old_start},{new_old_count} +{adjusted_new_start},{adjusted_new_count} @@{header_comment}"
					cumulative_blank_removals += blank_line_removals
				else:
					new_header = (
						f"@@ -{adjusted_old_start},{old_count} +{adjusted_new_start},{new_count} @@{header_comment}"
					)

				# For deleted files, always keep the hunks
				# For other files, only add hunks with valid changes
				if is_deleted_file or valid_changes:
					hunk_lines = [new_header] + hunk_body
					file_hunks.append(hunk_lines)
			else:
				# If we couldn't parse the header, add the hunk as is
				hunk_lines = [hunk_header]
				while i < len(lines) and not (lines[i].startswith("@@ ") or lines[i].startswith("diff ")):
					hunk_lines.append(lines[i])
					i += 1
				file_hunks.append(hunk_lines)

			# If we've reached the end of the file or the next file starts, add the file to the result
			if i >= len(lines) or lines[i].startswith("diff "):
				if file_hunks:  # Only add the file if it has at least one valid hunk
					result_lines.extend(file_header_lines)
					for hunk in file_hunks:
						result_lines.extend(hunk)
		else:
			i += 1

	return "\n".join(result_lines)


if __name__ == "__main__":
	try:
		patch_text = sys.stdin.read()
		result = process_patch(patch_text)
		print(result)
	except BrokenPipeError:
		# Handle case where output pipe is closed
		sys.stderr.close()
		sys.exit(1)
	except KeyboardInterrupt:
		# Handle Ctrl+C
		sys.stderr.write("\nInterrupted\n")
		sys.exit(1)