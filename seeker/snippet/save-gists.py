#date: 2024-02-21T17:00:33Z
#url: https://api.github.com/gists/2f7432a65519a01dd1ecf557f1b80d9b
#owner: https://api.github.com/users/mvandermeulen

"""
Save all gists from GitHub, including comments.

Requires the PyGithub library (``pip install PyGithub>=1.47``).

Pass a personal access token with the scope "gist" as the first
argument.

This probably won't work for large files (+10MiB), since they aren't
returned by the GitHub API, and must be cloned instead.
"""
import sys
from github import Github
from pathlib import Path

try:
    _prog, token = "**********"
except ValueError:
    print("usage: "**********"
    sys.exit(1)

gh = "**********"

base = Path.cwd()
user = gh.get_user()
info = [
    "# Gists",
    "",
]

for gist in user.get_gists():
    print("Gist", gist.id)
    gist_path = base / gist.id
    gist_path.mkdir(exist_ok=True)

    # index
    visibility = "" if gist.public else "🔒"
    gist_link = f"[{gist.id}]({gist.html_url})"
    info.extend([
        f"## {gist_link} {visibility}",
        "",
        f"Created: {gist.created_at.isoformat(timespec='seconds')}Z / "
        f"Updated: {gist.updated_at.isoformat(timespec='seconds')}Z",
        "",
        gist.description,
        "",
    ])
    info.extend([
        f"* `{file.filename}` - {file.language} - {file.last_modified}"
        for file in gist.files.values()
    ])
    info.append("")

    # comments
    if gist.comments:
        comments = [
            f"# Comments for {gist_link}",
            "",
        ]
        for comment in gist.get_comments():
            comments.extend([
                f"[{comment.id}]({gist.html_url}#gistcomment-{comment.id}) - "
                f"[{comment.user.login}]({comment.user.html_url}) - "
                f"Created: {comment.created_at.isoformat(timespec='seconds')}Z / ",
                f"Updated: {comment.updated_at.isoformat(timespec='seconds')}Z",
                "",
                comment.body,
                "",
            ])
        comment_path = gist_path / "comments.md"
        comment_path.write_text("\n".join(comments), encoding="utf-8")

    # files
    for file in gist.files.values():
        print("*", file.filename)
        file_path = gist_path / file.filename
        file_path.write_text(file.content, encoding="utf-8")

index_path = base / "index.md"
index_path.write_text("\n".join(info), encoding="utf-8")n(info), encoding="utf-8")