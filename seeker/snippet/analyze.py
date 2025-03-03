#date: 2025-03-03T17:09:33Z
#url: https://api.github.com/gists/6b990287c53507e2ae265fc888316205
#owner: https://api.github.com/users/aboutroots

#!/usr/bin/env python3
"""
Git Repository Analyzer

This script analyzes a git repository to provide insights into:
- File activity heatmap (age vs commit frequency)
- Architectural evolution
- Team dynamics
- Quality indicators
- Project health
- Feature development

Output is a markdown file.

Usage:
<from within a git repository root>
python analyze.py

Optional flags:
--dir <relative_path_from_repo_root> - Include only files from specific directory, e.g. frontend/
--ignore <relative_path_from_repo_root> - Exclude files from specific directory, e.g. backend/

Note that tests (.test.x, .spec.x etc) are ignored by default. The number of commits to look back in history is 2000.
"""

import argparse
import os
import sys
import subprocess
import re
import datetime
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
from dateutil.relativedelta import relativedelta

N_COMMITS = 2000


def run_command(command):
    """Run a shell command and return its output"""
    try:
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command timed out after 5 seconds: {command}")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error: {e}")
        print(f"STDERR: {e.stderr}")
        return ""


def validate_git_repo():
    """Check if the current directory is a git repository"""
    if not os.path.isdir(".git"):
        print("Error: The current directory is not a git repository.")
        print("Please run this script from the root of a git repository.")
        sys.exit(1)


def get_repo_name():
    """Get the repository name"""
    try:
        remote_url = run_command("git config --get remote.origin.url")
        if remote_url:
            # Extract repo name from URL
            repo_name = remote_url.split("/")[-1].split(".")[0]
            return repo_name
        else:
            return os.path.basename(os.getcwd())
    except:
        return os.path.basename(os.getcwd())


def get_first_commit_date():
    """Get the date of the first commit"""
    try:
        first_commit_date = run_command(
            "git log --reverse --date=short --format=%ad | head -1"
        )
        return first_commit_date
    except:
        return "unknown"


def get_repo_age():
    """Calculate the repository age in days"""
    try:
        first_commit = run_command("git log --reverse --format=%at | head -1")
        first_commit_timestamp = int(first_commit)
        now_timestamp = datetime.datetime.now().timestamp()
        age_seconds = now_timestamp - first_commit_timestamp
        age_days = age_seconds / (60 * 60 * 24)
        return int(age_days)
    except:
        return 0


def get_commit_count():
    """Get the total number of commits"""
    return run_command("git rev-list --count HEAD")


def get_contributors():
    """Get the list of contributors with commit counts"""
    contributors_raw = run_command(
        "git --no-pager shortlog -sne --no-merges HEAD"
    )
    contributors = []
    for line in contributors_raw.split("\n"):
        if line.strip():
            parts = re.match(r"^\s*(\d+)\s+(.*?)\s+<(.*)>$", line.strip())
            if parts:
                count, name, email = parts.groups()
                contributors.append(
                    {"name": name, "email": email, "commits": int(count)}
                )
    return contributors


def is_test_file(file_path):
    return any(
        [
            "test" in file_path.lower(),
            "spec" in file_path.lower(),
            "__tests__" in file_path.lower(),
            "mock" in file_path.lower(),
        ]
    )


def get_file_list(target_folder=None, ignored_path=None, ignore_tests=True):
    """
    Get all tracked files in the repository, optionally filtered by target folder
    """
    files_raw = run_command("git ls-files")
    all_files = files_raw.split("\n")

    # Filter by target folder if specified
    if target_folder:
        target_folder = target_folder.rstrip("/") + "/"
        all_files = [f for f in all_files if f.startswith(target_folder)]

    if ignored_path:
        ignored_path = ignored_path.rstrip("/") + "/"
        all_files = [f for f in all_files if not f.startswith(ignored_path)]

    if ignore_tests:
        all_files = [f for f in all_files if not is_test_file(f)]

    # Filter out non-source files and media/documentation files
    ignored_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".svg",
        ".ico",
        ".webp",
        ".mp4",
        ".webm",
        ".ogg",
        ".mp3",
        ".wav",
        ".flac",
        ".pdf",
        ".md",
        ".markdown",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".xls",
        ".xlsx",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
        ".css",
        ".scss",
        ".less",
        ".gitignore",
    }
    if ignore_tests:
        ignored_extensions.update(
            {".test", ".tests", ".spec", ".specs", ".feature"}
        )

    return [
        f
        for f in all_files
        if os.path.isfile(f)
        and os.path.splitext(f)[1].lower() not in ignored_extensions
    ]


def get_file_commit_history(file_path):
    """Get commit history for a specific file"""
    print(f"Running command git log -- '{file_path}'")
    commit_history = run_command(
        f"git log --format='%H,%an,%ae,%at,%s' -- '{file_path}'"
    )
    commits = []
    for line in commit_history.split("\n"):
        if line.strip():
            parts = line.strip().replace("'", "").split(",", 4)
            if len(parts) >= 5:
                hash, author, email, timestamp, message = parts
                commits.append(
                    {
                        "hash": hash,
                        "author": author,
                        "email": email,
                        "timestamp": int(timestamp),
                        "message": message,
                    }
                )
    return commits


def get_file_creation_date(file_path):
    """Get the creation date of a file in timestamp format"""
    print(
        "Running command git log --follow --format=%at --reverse --", file_path
    )
    creation_info = run_command(
        f"git log --follow --format=%at --reverse -- '{file_path}' | head -1"
    )
    if creation_info:
        return int(creation_info)
    return 0


def get_file_last_modified_date(file_path):
    """Get the last modified date of a file in timestamp format"""
    print(f"Running command git log -1 --format=%at -- '{file_path}'")
    modified_info = run_command(f"git log -1 --format=%at -- '{file_path}'")
    if modified_info:
        return int(modified_info)
    return 0


def get_file_extensions(files):
    """Get a count of file extensions in the repository"""
    extensions = Counter()
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext:
            extensions[ext] += 1
    return extensions


def get_directory_structure(files):
    """Get the directory structure and file counts"""
    dirs = defaultdict(int)
    for file in files:
        directory = os.path.dirname(file)
        if not directory:
            directory = "/"
        dirs[directory] += 1
    return dirs


def get_branch_info():
    """Get information about branches"""
    branches_raw = run_command("git branch -a")
    branches = [b.strip() for b in branches_raw.split("\n")]
    current_branch = run_command("git rev-parse --abbrev-ref HEAD")
    return {"all_branches": branches, "current_branch": current_branch}


def get_code_churn():
    """Calculate code churn - additions and deletions over time"""
    churn_raw = run_command(
        f"git --no-pager log -n {N_COMMITS} --pretty=format:'%ad' --date=short --numstat"
    )
    churn_by_date = defaultdict(
        lambda: {"additions": 0, "deletions": 0, "files": 0}
    )
    current_date = None

    for line in churn_raw.split("\n"):
        if line.startswith("'"):
            current_date = line.replace("'", "")
        elif line.strip() and current_date:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                additions, deletions = int(parts[0]), int(parts[1])
                churn_by_date[current_date]["additions"] += additions
                churn_by_date[current_date]["deletions"] += deletions
                churn_by_date[current_date]["files"] += 1

    return churn_by_date


def get_commit_message_quality():
    """Analyze commit message quality"""
    messages_raw = run_command(
        f"git --no-pager log -n {N_COMMITS} --pretty=format:'%s'"
    )
    messages = [msg.replace("'", "") for msg in messages_raw.split("\n")]

    quality = {
        "total": len(messages),
        "short": 0,  # < 10 chars
        "medium": 0,  # 10-50 chars
        "long": 0,  # > 50 chars
        "has_issue_ref": 0,  # Contains #123
        "has_merge_ref": 0,  # Merge references
        "has_fix_ref": 0,  # Contains "fix" or "hotfix"
    }

    for msg in messages:
        if len(msg) < 10:
            quality["short"] += 1
        elif len(msg) <= 50:
            quality["medium"] += 1
        else:
            quality["long"] += 1

        if re.search(r"#\d+", msg):
            quality["has_issue_ref"] += 1
        if "merge" in msg.lower():
            quality["has_merge_ref"] += 1
        if "fix" in msg.lower() or "hotfix" in msg.lower():
            quality["has_fix_ref"] += 1

    return quality


def get_test_coverage(files):
    """Analyze test coverage based on file paths"""
    test_files = [
        f for f in files if "test" in f.lower() or "spec" in f.lower()
    ]
    src_dirs = set()
    test_dirs = set()

    for file in files:
        directory = os.path.dirname(file)
        if "test" in directory.lower() or "spec" in directory.lower():
            test_dirs.add(directory)
        else:
            src_dirs.add(directory)

    return {
        "total_files": len(files),
        "test_files": len(test_files),
        "test_percentage": (
            round(len(test_files) / len(files) * 100, 2) if files else 0
        ),
        "src_dirs": len(src_dirs),
        "test_dirs": len(test_dirs),
    }


def get_orphaned_code(files):
    """Find potentially orphaned code (files not modified in a long time)"""
    orphaned = []

    now = datetime.datetime.now().timestamp()
    one_year_ago = now - (365 * 24 * 60 * 60)

    for file in files:
        last_mod = get_file_last_modified_date(file)
        if last_mod < one_year_ago:
            orphaned.append(
                {
                    "file": file,
                    "last_modified": datetime.datetime.fromtimestamp(
                        last_mod
                    ).strftime("%Y-%m-%d"),
                    "age_days": int((now - last_mod) / (24 * 60 * 60)),
                }
            )

    return orphaned


def get_recent_activity():
    """Get recent activity in the repository"""
    activity_raw = run_command(
        "git log --pretty=format:'%h,%an,%ad,%s' --date=short -n 20"
    )
    activity = []

    for line in activity_raw.split("\n"):
        if line.strip():
            parts = line.replace("'", "").split(",", 3)
            if len(parts) >= 4:
                hash, author, date, message = parts
                activity.append(
                    {
                        "hash": hash,
                        "author": author,
                        "date": date,
                        "message": message,
                    }
                )

    return activity


def get_commit_frequency():
    """Calculate commit frequency by day of week and hour"""
    day_raw = run_command(
        f"git --no-pager log -n {N_COMMITS} --pretty=format:'%ad' --date=format:'%u'"
    )
    hour_raw = run_command(
        f"git --no-pager log -n {N_COMMITS} --pretty=format:'%ad' --date=format:'%H'"
    )

    days = Counter([int(day) for day in day_raw.split("\n") if day.strip()])
    hours = Counter(
        [int(hour) for hour in hour_raw.split("\n") if hour.strip()]
    )

    return {"days": days, "hours": hours}


def analyze_single_file(file, now, six_months_ago):
    """Analyze a single file and return its data and category"""
    commits = get_file_commit_history(file)
    creation_date = get_file_creation_date(file)
    last_modified = get_file_last_modified_date(file)
    commit_count = len(commits)

    # Skip if no commits found
    if not commit_count:
        return None, None

    file_data = {
        "file": file,
        "creation_date": datetime.datetime.fromtimestamp(
            creation_date
        ).strftime("%Y-%m-%d"),
        "last_modified": datetime.datetime.fromtimestamp(
            last_modified
        ).strftime("%Y-%m-%d"),
        "commit_count": commit_count,
        "age_days": int((now - creation_date) / (24 * 60 * 60)),
        "recent_commits": sum(
            1 for c in commits if c["timestamp"] > six_months_ago
        ),
        "extension": os.path.splitext(file)[1].lower(),
    }

    # Categorize file based on age and activity
    is_recent = creation_date > six_months_ago
    is_active = commit_count > 5 or file_data["recent_commits"] > 3

    if is_recent and is_active:
        category = "recent_active"
    elif not is_recent and is_active:
        category = "old_active"
    elif is_recent and not is_active:
        category = "recent_stable"
    else:
        category = "old_stable"

    return file_data, category


def analyze_file_patterns(files, max_workers=None):
    """Analyze all files in parallel and categorize them by age and commit frequency"""
    now = datetime.datetime.now().timestamp()
    six_months_ago = now - (180 * 24 * 60 * 60)
    file_patterns = {
        "recent_active": [],  # Recent file, many commits
        "old_active": [],  # Old file, many commits
        "recent_stable": [],  # Recent file, few commits
        "old_stable": [],  # Old file, few commits
    }

    all_file_data = []

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Submit all file analysis tasks
        future_to_file = {
            executor.submit(
                analyze_single_file, file, now, six_months_ago
            ): file
            for file in files
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_data, category = future.result()
            if file_data is not None:
                all_file_data.append(file_data)
                file_patterns[category].append(file_data)

    return file_patterns, all_file_data


def create_heatmap_data(file_data):
    """Create data for the file activity heatmap visualization"""
    # Sort by file age (oldest to newest) and commit count (highest to lowest)
    sorted_data = sorted(
        file_data, key=lambda x: (x["age_days"], -x["commit_count"])
    )

    # Take top 50 files for better visualization
    top_files = sorted_data[:50]

    # Create heatmap data
    files = [os.path.basename(f["file"]) for f in top_files]
    age_days = [f["age_days"] for f in top_files]
    commit_count = [f["commit_count"] for f in top_files]

    # Normalize data for coloring
    max_age = max(age_days) if age_days else 1
    max_commits = max(commit_count) if commit_count else 1

    normalized_age = [a / max_age for a in age_days]
    normalized_commits = [c / max_commits for c in commit_count]

    return {
        "files": files,
        "age_days": age_days,
        "commit_count": commit_count,
        "normalized_age": normalized_age,
        "normalized_commits": normalized_commits,
    }


def generate_markdown_report(
    repo_name, target_folder, ignored_folder, analysis_results
):
    """Generate the comprehensive markdown report"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title_suffix = f"/{target_folder}" if target_folder else ""
    subtitle = f"(excluding {ignored_folder})" if ignored_folder else ""

    report = [
        f"# Git Repository Analysis: {repo_name}{title_suffix}",
        subtitle,
        f"**Generated on:** {now}",
        "\n## 📊 Repository Overview",
        f"* **Age:** {analysis_results['repo_age']} days (since {analysis_results['first_commit_date']})",
        f"* **Total Commits:** {analysis_results['commit_count']}",
        f"* **Contributors:** {len(analysis_results['contributors'])}",
        f"* **Files:** {len(analysis_results['file_list'])}",
        f"* **Current Branch:** {analysis_results['branch_info']['current_branch']}",
        "\n## 📈 Activity Heatmap (File Age vs. Commit Frequency)",
        "This section categorizes files based on their age and commit activity:",
        "\n### 🔥 Recent & Active Files (New features, active development)",
        f"*{len(analysis_results['file_patterns']['recent_active'])} files*",
    ]

    # Add top recent & active files
    top_files = sorted(
        analysis_results["file_patterns"]["recent_active"],
        key=lambda x: x["commit_count"],
        reverse=True,
    )[:20]
    if top_files:
        report.append("| File | Age (days) | Commits | Last Modified |")
        report.append("|------|------------|---------|---------------|")
        for file in top_files:
            report.append(
                f"| {file['file']} | {file['age_days']} | {file['commit_count']} | {file['last_modified']} |"
            )
    else:
        report.append("*No files in this category*")

    report.extend(
        [
            "\n### ⚠️ Old & Active Files (Possible trouble spots, important components)",
            f"*{len(analysis_results['file_patterns']['old_active'])} files*",
        ]
    )

    # Add top old & active files
    top_files = sorted(
        analysis_results["file_patterns"]["old_active"],
        key=lambda x: x["recent_commits"],
        reverse=True,
    )[:20]
    if top_files:
        report.append(
            "| File | Age (days) | Total Commits | Recent Commits | Last Modified |"
        )
        report.append(
            "|------|------------|---------------|----------------|---------------|"
        )
        for file in top_files:
            report.append(
                f"| {file['file']} | {file['age_days']} | {file['commit_count']} | {file['recent_commits']} | {file['last_modified']} |"
            )
    else:
        report.append("*No files in this category*")

    report.extend(
        [
            "\n### 🆕 Recent & Stable Files (New but less important or well-written code)",
            f"*{len(analysis_results['file_patterns']['recent_stable'])} files*",
        ]
    )

    # Add top recent & stable files
    top_files = sorted(
        analysis_results["file_patterns"]["recent_stable"],
        key=lambda x: x["age_days"],
    )[:20]
    if top_files:
        report.append("| File | Age (days) | Commits | Last Modified |")
        report.append("|------|------------|---------|---------------|")
        for file in top_files:
            report.append(
                f"| {file['file']} | {file['age_days']} | {file['commit_count']} | {file['last_modified']} |"
            )
    else:
        report.append("*No files in this category*")

    report.extend(
        [
            "\n### 🏛️ Old & Stable Files (Core, stable components)",
            f"*{len(analysis_results['file_patterns']['old_stable'])} files*",
        ]
    )

    # Add top old & stable files (oldest first)
    top_files = sorted(
        analysis_results["file_patterns"]["old_stable"],
        key=lambda x: x["age_days"],
        reverse=True,
    )[:20]
    if top_files:
        report.append("| File | Age (days) | Commits | Last Modified |")
        report.append("|------|------------|---------|---------------|")
        for file in top_files:
            report.append(
                f"| {file['file']} | {file['age_days']} | {file['commit_count']} | {file['last_modified']} |"
            )
    else:
        report.append("*No files in this category*")

    # Add architectural evolution section
    report.extend(
        [
            "\n## 🏗️ Architectural Evolution",
            "Understanding how the project structure has evolved over time.",
        ]
    )

    # Add file extension distribution
    report.append("\n### File Type Distribution")
    extensions = analysis_results["file_extensions"]
    if extensions:
        report.append("| Extension | Count | Percentage |")
        report.append("|-----------|-------|------------|")
        total = sum(extensions.values())
        for ext, count in extensions.most_common(10):
            percentage = round(count / total * 100, 2)
            report.append(f"| {ext} | {count} | {percentage}% |")
    else:
        report.append("*No file extension data available*")

    # Add directory structure information
    report.append("\n### Directory Structure")
    dirs = analysis_results["directory_structure"]
    if dirs:
        report.append("| Directory | Files |")
        report.append("|-----------|-------|")
        for dir_name, count in sorted(
            dirs.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            report.append(f"| {dir_name} | {count} |")
    else:
        report.append("*No directory structure data available*")

    # Add team dynamics section
    report.extend(
        [
            "\n## 👥 Team Dynamics",
            "Understanding how team members collaborate on the codebase.",
        ]
    )

    # Add contributor information
    report.append("\n### Top Contributors")
    contributors = analysis_results["contributors"]
    if contributors:
        report.append("| Contributor | Commits | Percentage |")
        report.append("|-------------|---------|------------|")
        total_commits = sum(c["commits"] for c in contributors)
        for contributor in sorted(
            contributors, key=lambda x: x["commits"], reverse=True
        )[:20]:
            percentage = round(contributor["commits"] / total_commits * 100, 2)
            report.append(
                f"| {contributor['name']} | {contributor['commits']} | {percentage}% |"
            )
    else:
        report.append("*No contributor data available*")

    # Add commit frequency information
    report.append("\n### Commit Patterns")
    frequency = analysis_results["commit_frequency"]

    # Days of week
    days_map = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }
    if frequency["days"]:
        report.append("\n**Commits by Day of Week**")
        report.append("| Day | Commits | Percentage |")
        report.append("|-----|---------|------------|")
        total_day_commits = sum(frequency["days"].values())
        for day_num in range(1, 8):
            day_name = days_map[day_num]
            count = frequency["days"][day_num]
            percentage = (
                round(count / total_day_commits * 100, 2)
                if total_day_commits
                else 0
            )
            report.append(f"| {day_name} | {count} | {percentage}% |")

    # Hours of day
    if frequency["hours"]:
        report.append("\n**Commits by Hour of Day**")
        report.append("| Hour | Commits | Percentage |")
        report.append("|------|---------|------------|")
        total_hour_commits = sum(frequency["hours"].values())
        for hour in range(24):
            count = frequency["hours"][hour]
            percentage = (
                round(count / total_hour_commits * 100, 2)
                if total_hour_commits
                else 0
            )
            report.append(f"| {hour:02d}:00 | {count} | {percentage}% |")

    # Add quality indicators section
    report.extend(
        [
            "\n## 🔍 Quality Indicators",
            "Metrics that can indicate code quality and maintenance practices.",
        ]
    )

    # Add commit message quality
    quality = analysis_results["commit_message_quality"]
    if quality:
        report.append("\n### Commit Message Quality")
        total = quality["total"]
        report.append(f"* **Total Commits:** {total}")
        report.append(
            f"* **Short Messages (<10 chars):** {quality['short']} ({round(quality['short']/total*100, 2)}%)"
        )
        report.append(
            f"* **Medium Messages (10-50 chars):** {quality['medium']} ({round(quality['medium']/total*100, 2)}%)"
        )
        report.append(
            f"* **Long Messages (>50 chars):** {quality['long']} ({round(quality['long']/total*100, 2)}%)"
        )
        report.append(
            f"* **Messages with Issue References:** {quality['has_issue_ref']} ({round(quality['has_issue_ref']/total*100, 2)}%)"
        )
        report.append(
            f"* **Messages with Fix References:** {quality['has_fix_ref']} ({round(quality['has_fix_ref']/total*100, 2)}%)"
        )

    # Add test coverage information
    coverage = analysis_results["test_coverage"]
    if coverage:
        report.append("\n### Test Coverage Indicators")
        report.append(f"* **Total Files:** {coverage['total_files']}")
        report.append(
            f"* **Test Files:** {coverage['test_files']} ({coverage['test_percentage']}%)"
        )
        report.append(f"* **Source Directories:** {coverage['src_dirs']}")
        report.append(f"* **Test Directories:** {coverage['test_dirs']}")

    # Add project health section
    report.extend(
        [
            "\n## 🏥 Project Health",
            "Overall indicators of project activity and maintenance status.",
        ]
    )

    # Add code churn information
    churn = analysis_results["code_churn"]
    if churn:
        report.append("\n### Recent Code Churn")
        report.append(
            "| Date | Files Changed | Additions | Deletions | Net Change |"
        )
        report.append(
            "|------|---------------|-----------|-----------|------------|"
        )

        # Get last 10 dates with activity
        recent_dates = sorted(churn.keys(), reverse=True)[:10]
        for date in recent_dates:
            data = churn[date]
            net = data["additions"] - data["deletions"]
            sign = "+" if net >= 0 else ""
            report.append(
                f"| {date} | {data['files']} | +{data['additions']} | -{data['deletions']} | {sign}{net} |"
            )

    # Add orphaned code information
    orphaned = analysis_results["orphaned_code"]
    if orphaned:
        report.append("\n### Potentially Orphaned Code")
        report.append("Files not modified in over a year:")
        report.append("| File | Last Modified | Age (days) |")
        report.append("|------|---------------|------------|")
        for file in sorted(
            orphaned, key=lambda x: x["age_days"], reverse=True
        )[:10]:
            report.append(
                f"| {file['file']} | {file['last_modified']} | {file['age_days']} |"
            )

    # Add feature development section
    report.extend(
        [
            "\n## 🚀 Feature Development",
            "Information about feature development cycles and recent activity.",
        ]
    )

    # Add recent activity
    activity = analysis_results["recent_activity"]
    if activity:
        report.append("\n### Recent Activity")
        report.append("| Date | Author | Commit | Message |")
        report.append("|------|--------|--------|---------|")
        for commit in activity:
            report.append(
                f"| {commit['date']} | {commit['author']} | {commit['hash']} | {commit['message']} |"
            )

    # Add conclusion
    report.extend(
        [
            "\n## 🔮 Conclusion & Recommendations",
            "Based on the repository analysis, here are some key insights:",
            "",
            f"1. **Primary Focus Areas:** Look at the {len(analysis_results['file_patterns']['recent_active'])} files in the 'Recent & Active' category first, as they represent current development priorities.",
            "",
            f"2. **Potential Trouble Spots:** Review the {len(analysis_results['file_patterns']['old_active'])} files in the 'Old & Active' category, as frequent changes to old code may indicate technical debt or critical components.",
            "",
            f"3. **Code Examples:** For learning how things are done in this codebase, review the {len(analysis_results['file_patterns']['recent_stable'])} files in the 'Recent & Stable' category for clean, newer implementations.",
            "",
            f"4. **Stable Core:** The {len(analysis_results['file_patterns']['old_stable'])} files in the 'Old & Stable' category represent the mature, well-tested codebase that rarely changes.",
            "",
            "5. **Team Collaboration:** Identify the main contributors and reach out to them for domain-specific knowledge.",
            "",
            "6. **Code Health:** Pay attention to test coverage indicators and potential orphaned code areas.",
            "",
            "7. **Next Steps:**",
            "   * Start by exploring the most active directories",
            "   * Review recent commits to understand current focus",
            "   * Identify core components by analyzing old but stable files",
            "   * Contact top contributors for specialized knowledge",
        ]
    )

    return "\n".join(report)


def main():
    print("Git Repository Analyzer")
    print("----------------------")
    print("Checking repository...")

    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument(
        "--dir",
        type=str,
        required=False,
        default=None,
        help="Path to specific directory",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        required=False,
        default=None,
        help="Path to a directory to ignore",
    )
    parser.add_argument(
        "--ignore-tests",
        type=bool,
        required=False,
        default=True,
        help="If true, will ignore test files in the analysis",
    )
    args = parser.parse_args()

    validate_git_repo()
    repo_name = get_repo_name()

    print(f"Analyzing repository: {repo_name}")
    print("This may take some time depending on repository size...")

    # Collect all analysis data
    analysis_results = {}

    # Basic repo info
    analysis_results["repo_name"] = repo_name
    analysis_results["first_commit_date"] = get_first_commit_date()
    analysis_results["repo_age"] = get_repo_age()
    analysis_results["commit_count"] = get_commit_count()
    analysis_results["contributors"] = get_contributors()

    files = get_file_list(
        target_folder=args.dir,
        ignored_path=args.ignore,
        ignore_tests=args.ignore_tests,
    )
    analysis_results["file_list"] = files
    analysis_results["branch_info"] = get_branch_info()

    print("Analyzing file patterns...")
    analysis_results["file_patterns"], all_file_data = analyze_file_patterns(
        files
    )
    analysis_results["heatmap_data"] = create_heatmap_data(all_file_data)

    print("Collecting architectural information...")
    analysis_results["file_extensions"] = get_file_extensions(files)
    analysis_results["directory_structure"] = get_directory_structure(files)

    print("Analyzing code quality indicators...")
    analysis_results["commit_message_quality"] = get_commit_message_quality()
    if not args.ignore_tests:
        analysis_results["test_coverage"] = get_test_coverage(files)
    else:
        analysis_results["test_coverage"] = None

    print("Analyzing project health...")
    analysis_results["code_churn"] = get_code_churn()
    analysis_results["orphaned_code"] = get_orphaned_code(files)

    print("Analyzing development patterns...")
    analysis_results["recent_activity"] = get_recent_activity()
    analysis_results["commit_frequency"] = get_commit_frequency()

    # Generate report
    print("Generating markdown report...")
    report = generate_markdown_report(
        repo_name, args.dir, args.ignore, analysis_results
    )

    # Save report
    output_file = f"{repo_name}-git-analysis.md"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nAnalysis complete! Report saved to: {output_file}")


if __name__ == "__main__":
    main()
