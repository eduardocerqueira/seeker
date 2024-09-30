#date: 2024-09-30T17:03:15Z
#url: https://api.github.com/gists/ebeb9f472b2e53950a9029ad65ae4a2f
#owner: https://api.github.com/users/Joel-hanson

import argparse
import json
import re
import sys
from datetime import datetime

import frontmatter


def convert_markdown_to_medium(input_file, output_file):
    # Read the markdown file
    with open(input_file, "r", encoding="utf-8") as file:
        post = frontmatter.load(file)

    # Extract front matter
    title = post.get("title", "")
    tags = post.get("tags", [])
    canonical_url = post.get("canonical_url", "")
    published_at = post.get("date")

    # Convert published_at to ISO format if it exists
    if published_at:
        published_at = datetime.fromisoformat(str(published_at)).isoformat()

    # Get the content without front matter
    content = post.content

    # Convert headers
    content = re.sub(r"^# (.*?)$", r"# \1", content, flags=re.MULTILINE)
    content = re.sub(r"^## (.*?)$", r"## \1", content, flags=re.MULTILINE)
    content = re.sub(r"^### (.*?)$", r"### \1", content, flags=re.MULTILINE)

    # Convert links
    content = re.sub(r"\[(.*?)\]\((.*?)\)", r"[\1](\2)", content)

    # Convert code blocks
    content = re.sub(
        r"```(\w+)?\n(.*?)\n```", r"```\1\n\2\n```", content, flags=re.DOTALL
    )

    # Convert images
    content = re.sub(r"!\[(.*?)\]\((.*?)\)", r"![\1](\2)", content)

    # Create the Medium import format
    medium_post = {
        "title": title,
        "content": content,
        "tags": tags,
        "canonicalUrl": canonical_url,
        "publishStatus": "draft",
    }

    if published_at:
        medium_post["publishedAt"] = published_at

    # Write the JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(medium_post, file, ensure_ascii=False, indent=2)

    print(f"Conversion complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown to Medium format")
    parser.add_argument("input_file", help="Input Markdown file")
    parser.add_argument("output_file", help="Output JSON file for Medium")
    args = parser.parse_args()

    convert_markdown_to_medium(args.input_file, args.output_file)
