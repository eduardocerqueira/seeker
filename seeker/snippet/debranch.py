#date: 2025-08-28T17:12:30Z
#url: https://api.github.com/gists/d28ad55e0097d502f8cbc5d7449d12f9
#owner: https://api.github.com/users/tbnorth

#!/usr/bin/env python3
"""Local repo cleanup"""
import re
import sys
from subprocess import run

against = sys.argv[1] if len(sys.argv) > 1 else "dev"
remote = sys.argv[2] if len(sys.argv) > 2 else None


def do_cmd(cmd, silent=False):
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    if not silent:
        print(cmd)
    proc = run(cmd, shell=True, capture_output=True)
    if proc.stderr:
        print(proc.stderr.decode("utf8"))
    return proc.stdout.decode("utf8")


COMMITS = do_cmd("git log --all --pretty=format:'%h %ad %s'").splitlines()


def as_list(text: str) -> list[str]:
    """List of lines (\n) in text."""
    return [i.strip() for i in text.strip().split("\n")]


def dev_diff(branch):
    """Try a merge without committing to detect differences."""
    # https://stackoverflow.com/a/36439978
    do_cmd(f"git merge --no-commit {branch}")
    diff = bool(do_cmd(f"git diff --stat {against} | head"))
    print(f"{'NO difference' if not diff else 'Differences'} seen")
    do_cmd("git merge --abort")
    return diff


# Get list of branches to consider.
if remote:
    git_branch = f"git branch -r {{merge_target}} | grep '^  {remote}'"
    prefix = f"^{remote}/"
else:
    git_branch = "git branch {merge_target}"
    prefix = ""
branches = as_list(do_cmd(git_branch.format(merge_target="")))

# Assume branches with simple names are permanent.
permanent = [re.sub(prefix, "", i.strip("* ")) for i in branches]
permanent = [i for i in permanent if not re.search("[-/_]", i) or i.startswith("*")]
if remote:
    permanent = [remote + "/" + i for i in permanent]

cmd = [git_branch.format(merge_target=" ".join(f"--merged {i}" for i in permanent))]
merged = [i for i in as_list(do_cmd(cmd)) if i not in permanent and "*" not in i]

cmd = [git_branch.format(merge_target=" ".join(f"--no-merged {i}" for i in permanent))]
unmerged = [i for i in as_list(do_cmd(cmd)) if i not in permanent and "*" not in i]


def old_no_diff():
    no_diff = []
    do_cmd(f"git checkout {against}")
    for branch in unmerged:
        print(f"\nChecking for differences in {branch}")
        if not dev_diff(branch):
            no_diff.append(branch)


def check_diff(against="dev"):
    """For "unmerged" branches, check if there is a commit with same message *and
    timestamp* in the permanent branch.
    """
    no_diff = []
    for branch in unmerged:
        # Get msg. and timestamp of last commit in branch
        commit = do_cmd(f"git log -1 --pretty=format:'%h %ad %s' {branch}")
        print("\nis", commit)
        branch_hash = commit[:7]
        print(f"\nChecking for differences in {branch_hash} {branch}")
        any_merged = False
        for line in COMMITS:
            # Look for same msg. and timestamp in permanent branch
            print("in", line)
            if commit[8:] in line:
                hash = line[:7]
                merged = do_cmd(
                    f"git merge-base --is-ancestor {hash} {against} "
                    "&& echo -n ' IN' || echo -n 'OUT'",
                    silent=True,
                )
                any_merged = any_merged or "IN" in merged
                print(f"{hash} {merged} {line[8:]}")
        if any_merged:
            no_diff.append(branch)

    return no_diff


no_diff = check_diff(against)
if remote:
    drop_branch = "git push --delete"
    ref = lambda x: x.replace("/", " ")
else:
    drop_branch = "git branch -D"
    ref = lambda x: x

print("\nPermanent:", " ".join(permanent))
print(
    "Merged:" + ("\n " if merged else ""),
    "\n  ".join(f"{drop_branch.lower()} {ref(i)}" for i in merged),
)
print("Unmerged:", " ".join(unmerged))
print(
    f"Unmerged but in {against}:" + ("\n " if no_diff else ""),
    "\n  ".join(f"{drop_branch} {ref(i)}" for i in no_diff if i != against),
)
not_in = set(unmerged) - set(no_diff)
print(f"Unmerged and not in {against}:")
for branch in not_in:
    time_text = do_cmd(f"git log -1 --pretty=format:'%ad %s' {branch}", silent=True)
    branches = set()
    for line in COMMITS:
        if time_text in line:
            hash = line[:7]
            contained_in = do_cmd("git branch --contains " + hash, silent=True)
            branches |= {
                i.strip("* ")
                for i in contained_in.splitlines()
                if i.strip("* ") != branch
            }

    if branches:
        print(f"  {branch}  # in: {' '.join(branches)}")
    else:
        print(f"  {branch}")
