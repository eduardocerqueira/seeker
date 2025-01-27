#date: 2025-01-27T17:06:28Z
#url: https://api.github.com/gists/11c2ca56bc3fe4b6ada868312125c322
#owner: https://api.github.com/users/ameen-jd1

import os
import re
import subprocess
from colorama import Fore, init, Style

# Initialize colorama
init(autoreset=True)

def validate_file(file_path):
    """Validate if the file exists and is a Python file."""
    if not os.path.isfile(file_path) or not file_path.endswith(".py"):
        print(Fore.RED + "Error: Please provide a valid Python file path.")
        return False
    return True


def run_subprocess(command, success_msg, error_msg):
    """Run a subprocess command and handle errors."""
    try:
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            return result.stdout
        print(Fore.GREEN + success_msg)
        return None
    except Exception as e:
        print(Fore.RED + f"{error_msg}: {e}")
        return None


def parse_pylint_output(output):
    """Parse pylint output and classify issues as major or minor."""
    major_keywords = ["error", "undefined-variable", "unused-variable", "unused-import"]
    major_issues = set()
    minor_issues = set()

    for line in output.splitlines():
        if any(keyword in line.lower() for keyword in major_keywords):
            major_issues.add(Fore.RED + line)
        else:
            minor_issues.add(Fore.YELLOW + line)

    return major_issues, minor_issues


def display_issues(major_issues, minor_issues):
    """Display major and minor issues."""
    if major_issues:
        print(Fore.RED + "\nMajor Issues:")
        for issue in sorted(major_issues):
            print(issue)

    if minor_issues:
        print(Fore.YELLOW + "\nMinor Issues:")
        for issue in sorted(minor_issues):
            print(issue)


def check_prohibited_patterns(file_path):
    """Check for prohibited patterns in the file."""
    prohibited_patterns = {
    "time.sleep()": r"time\.sleep\s*\([\s\S]*?\)",
    ".pause()": r"\.\b\w*pause\s*\([\s\S]*?\)",
    }

    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            content = f.read()
            for name, pattern in prohibited_patterns.items():
                if re.search(pattern, content):
                    print(Fore.RED + f"Prohibited usage found: '{name}' in {file_path}")
    except Exception as e:
        print(Fore.RED + f"Error reading the file: {e}")

def pylint_check(file_path):
    """
    Check for code issues using pylint.
    """
    print("\n*** Checking for issues with pylint ***")
    pylint_output = run_subprocess(
        ["pylint", file_path],
        "No issues detected with pylint.",
        "Error running pylint"
    )
    if pylint_output:
        major_issues, minor_issues = parse_pylint_output(pylint_output)

        # Remove the "Your code has been rated" line from minor issues
        minor_issues = {issue for issue in minor_issues if "Your code has been rated" not in issue}

        display_issues(major_issues, minor_issues)

        # Find the line with the code rating and print it in blue, bold, and large size at the end
        for line in pylint_output.splitlines():
            if "Your code has been rated" in line:
                print(Fore.BLUE + Style.BRIGHT + line)  # For blue and bold text  # Simulated bold, blue, and large size with ANSI codes


def check_code_quality(file_path):
    """Run quality checks on a Python file."""
    if not validate_file(file_path):
        return

    print(Fore.GREEN + f"\n*** Starting quality checks for {file_path} ***")

    # Check for issues with pylint
    pylint_check(file_path)

    # Check for prohibited patterns
    print("\n*** Checking for prohibited usage of 'time.sleep()' and '.pause()' ***")

    check_prohibited_patterns(file_path)

    print(Fore.GREEN + "\n*** All checks completed ***")

if __name__ == "__main__":
    # Update the file path to your Python file
    file_path = os.path.abspath("expensify_test.py")  # Replace with your file path
    check_code_quality(file_path)