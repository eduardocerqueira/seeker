#date: 2025-07-21T17:08:36Z
#url: https://api.github.com/gists/cce5451ab00183e7dfeef5cc31ffefbe
#owner: https://api.github.com/users/AngeloR

import feedparser
import os
import re
import frontmatter
from markdownify import markdownify as md
import html
from datetime import datetime

RSS_URL = "https://medium.com/feed/@xangelo"
OUTPUT_DIR = "content/posts/medium"
EXISTING_SLUGS = {f[:-3] for f in os.listdir(OUTPUT_DIR) if f.endswith(".md")}

def slugify(title):
    return re.sub(r"[^\w-]", "", re.sub(r"\s+", "-", title.lower())).strip("-")

feed = feedparser.parse(RSS_URL)

for entry in feed.entries:
    slug = slugify(entry.title)
    if slug in EXISTING_SLUGS:
        continue

    content_html = entry.get("content", [{}])[0].get("value", "") or entry.get("summary", "")
    markdown_content = md(html.unescape(content_html))

    post = frontmatter.Post(markdown_content)
    post["title"] = entry.title
    post["date"] = entry.published
    post["slug"] = slug
    post["draft"] = False
    post["medium_link"] = entry.link

    output_path = os.path.join(OUTPUT_DIR, f"{slug}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    print(f"Saved: {output_path}")