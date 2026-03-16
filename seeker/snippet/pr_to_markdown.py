#date: 2026-03-16T17:38:22Z
#url: https://api.github.com/gists/69ea374f30c1a78c97531bfc4b7c3462
#owner: https://api.github.com/users/0xntpower

"""
PR Review Comments to Markdown Exporter

Dumps all review comments from a GitHub PR into a structured markdown file
that can be fed to Claude Code for assistance in implementing fixes.

Requirements:
    - GitHub CLI (gh) installed and authenticated
    - Python 3.7+

Usage:
    python pr_to_markdown.py OWNER/REPO PR_NUMBER [OUTPUT_FILE]

Example:
    python pr_to_markdown.py githubusername/mypr prnum
    python pr_to_markdown.py 0xntpower/netwatch 115 review.md
"""

import subprocess
import json
import sys
import os
from datetime import datetime


def run_gh(args: list[str]) -> str:
    """Run a gh CLI command and return stdout."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"[!] gh command failed: gh {' '.join(args)}", file=sys.stderr)
        print(f"    stderr: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout


def gh_api_paginate(endpoint: str) -> list[dict]:
    """Fetch all pages from a GitHub API endpoint via gh."""
    raw = run_gh(["api", endpoint, "--paginate"])
    if not raw.strip():
        return []

    # --paginate concatenates JSON arrays, so we may get `][` between pages
    raw = raw.strip()
    if raw.startswith("["):
        raw = raw.replace("][", ",")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"[!] Failed to parse JSON from: {endpoint}", file=sys.stderr)
        return []


def fetch_pr_details(repo: str, pr: int) -> dict:
    """Fetch PR title, body, state, head branch, etc."""
    raw = run_gh([
        "pr", "view", str(pr),
        "--repo", repo,
        "--json", "title,body,state,headRefName,baseRefName,author,url"
    ])
    if not raw.strip():
        return {}
    return json.loads(raw)


def fetch_reviews(repo: str, pr: int) -> list[dict]:
    """Fetch review summaries (approve/request changes/comment)."""
    return gh_api_paginate(f"repos/{repo}/pulls/{pr}/reviews")


def fetch_review_comments(repo: str, pr: int) -> list[dict]:
    """Fetch inline code review comments."""
    return gh_api_paginate(f"repos/{repo}/pulls/{pr}/comments")


def fetch_issue_comments(repo: str, pr: int) -> list[dict]:
    """Fetch general conversation comments."""
    return gh_api_paginate(f"repos/{repo}/issues/{pr}/comments")


def fetch_diff(repo: str, pr: int) -> str:
    """Fetch the PR diff."""
    return run_gh(["pr", "diff", str(pr), "--repo", repo])


def build_markdown(
    pr_details: dict,
    reviews: list[dict],
    review_comments: list[dict],
    issue_comments: list[dict],
    diff: str,
    repo: str,
    pr: int,
) -> str:
    lines: list[str] = []

    # --- Header ---
    title = pr_details.get("title", f"PR #{pr}")
    url = pr_details.get("url", "")
    author = pr_details.get("author", {}).get("login", "unknown")
    state = pr_details.get("state", "unknown")
    head = pr_details.get("headRefName", "?")
    base = pr_details.get("baseRefName", "?")

    lines.append(f"# PR #{pr}: {title}")
    lines.append("")
    lines.append(f"- **Repository:** `{repo}`")
    lines.append(f"- **Author:** {author}")
    lines.append(f"- **State:** {state}")
    lines.append(f"- **Branch:** `{head}` → `{base}`")
    if url:
        lines.append(f"- **URL:** {url}")
    lines.append(f"- **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # --- PR Body ---
    body = pr_details.get("body", "").strip()
    if body:
        lines.append("## PR Description")
        lines.append("")
        lines.append(body)
        lines.append("")

    # --- Review Summaries ---
    reviews_with_body = [r for r in reviews if r.get("body", "").strip()]
    if reviews_with_body:
        lines.append("## Review Summaries")
        lines.append("")
        for r in reviews_with_body:
            user = r.get("user", {}).get("login", "unknown")
            state = r.get("state", "COMMENTED")
            rbody = r.get("body", "").strip()
            lines.append(f"### {user} ({state})")
            lines.append("")
            lines.append(rbody)
            lines.append("")
            lines.append("---")
            lines.append("")

    # --- Inline Code Review Comments ---
    if review_comments:
        lines.append("## Inline Code Review Comments")
        lines.append("")

        # Group by file
        by_file: dict[str, list[dict]] = {}
        for c in review_comments:
            path = c.get("path", "unknown")
            by_file.setdefault(path, []).append(c)

        for path, comments in sorted(by_file.items()):
            lines.append(f"### `{path}`")
            lines.append("")
            for c in sorted(comments, key=lambda x: x.get("line") or x.get("original_line") or 0):
                user = c.get("user", {}).get("login", "unknown")
                line = c.get("line") or c.get("original_line") or "?"
                cbody = c.get("body", "").strip()
                diff_hunk = c.get("diff_hunk", "").strip()

                lines.append(f"**{user}** (line {line}):")
                lines.append("")
                if diff_hunk:
                    lines.append("<details><summary>Code context</summary>")
                    lines.append("")
                    lines.append("```diff")
                    lines.append(diff_hunk)
                    lines.append("```")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                lines.append(cbody)
                lines.append("")
                lines.append("---")
                lines.append("")

    # --- Conversation Comments ---
    if issue_comments:
        lines.append("## Conversation Comments")
        lines.append("")
        for c in issue_comments:
            user = c.get("user", {}).get("login", "unknown")
            cbody = c.get("body", "").strip()
            created = c.get("created_at", "")[:10]
            lines.append(f"**{user}** ({created}):")
            lines.append("")
            lines.append(cbody)
            lines.append("")
            lines.append("---")
            lines.append("")

    # --- Diff ---
    if diff.strip():
        lines.append("## Full Diff")
        lines.append("")
        lines.append("```diff")
        lines.append(diff.strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} OWNER/REPO PR_NUMBER [OUTPUT_FILE]")
        print(f"Example: {sys.argv[0]} 0xntpower/netwatch 115")
        sys.exit(1)

    repo = sys.argv[1]
    pr = int(sys.argv[2])
    output = sys.argv[3] if len(sys.argv) > 3 else f"pr_{pr}_review.md"

    print(f"[*] Fetching PR #{pr} from {repo}...")

    print("    -> PR details...")
    pr_details = fetch_pr_details(repo, pr)

    print("    -> Review summaries...")
    reviews = fetch_reviews(repo, pr)

    print("    -> Inline code comments...")
    review_comments = fetch_review_comments(repo, pr)

    print("    -> Conversation comments...")
    issue_comments = fetch_issue_comments(repo, pr)

    print("    -> Diff...")
    diff = fetch_diff(repo, pr)

    # Stats
    total_comments = len(reviews) + len(review_comments) + len(issue_comments)
    print(f"[*] Found {total_comments} total comments:")
    print(f"    - {len(reviews)} review summaries")
    print(f"    - {len(review_comments)} inline code comments")
    print(f"    - {len(issue_comments)} conversation comments")

    md = build_markdown(pr_details, reviews, review_comments, issue_comments, diff, repo, pr)

    with open(output, "w", encoding="utf-8") as f:
        f.write(md)

    size_kb = os.path.getsize(output) / 1024
    print(f"[+] Written to {output} ({size_kb:.1f} KB)")
    print()
    print("Now run Claude Code:")
    print(f'  claude "Read {output} for all CR feedback and implement the fixes. '
          f'Use gh and git as needed for additional context."')


if __name__ == "__main__":
    main()
