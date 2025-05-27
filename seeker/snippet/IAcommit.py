#date: 2025-05-27T17:13:59Z
#url: https://api.github.com/gists/648aed896e798e308882d9cda88d598d
#owner: https://api.github.com/users/Hammer2900

"""
usage: IAcommit.py [-h] [--limit LIMIT] [--model MODEL] [--hash HASH] repository_path

Analyzes commits in a Git repository and suggests new messages using AI with Ollama.

Positional arguments:
  repository_path       The local directory path of the Git repository to analyze.

Optional arguments:
  -h, --help            show this help message and exit
  --limit LIMIT         Maximum number of recent commits to analyze (optional).
                        If not specified, all commits will be analyzed (this may
                        take a long time). This is ignored if --hash is specified.
  --model MODEL         Name of the Ollama model to use (e.g., llama3, mistral,
                        codellama, tavernari/git-commit-message).
                        Default: llama3. Ensure it has been pulled with
                        'ollama pull <model_name>'.
  --hash HASH           The specific commit hash to analyze (optional).
                        If specified, only this commit will be analyzed.

Examples:
  # Analyze all commits in './my-repo' using the default 'llama3' model
  python IAcommit.py ./my-repo

  # Analyze the last 5 commits in '/path/to/another/repo' using the 'mistral' model
  python IAcommit.py /path/to/another/repo --limit 5 --model mistral

  # Analyze a specific commit 'a1b2c3d' in './my-repo'
  python IAcommit.py ./my-repo --hash a1b2c3d

  # Analyze a specific commit using 'codellama' model
  python IAcommit.py "./my project repo" --hash a1b2c3d --model codellama
"""
import subprocess
import ollama
import argparse
import os
import sys

def check_if_git_repo(path):
    if not os.path.isdir(path):
        print(f"Error: The specified path '{path}' is not a directory.")
        return False
    git_dir = os.path.join(path, '.git')
    if not os.path.isdir(git_dir):
        print(f"Error: The directory '{path}' does not appear to be a Git repository (missing .git folder).")
        return False
    return True

def get_commit_hashes(limit=None):
    command = ['git', 'log', '--pretty=format:%H']
    if limit:
        command.append(f'-n {limit}')
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return [line for line in result.stdout.strip().split('\n') if line]
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit hashes: {e}")
        print(f"Stderr output: {e.stderr}")
        return []
    except FileNotFoundError:
        print("Error: Git does not seem to be installed or in the PATH.")
        return []

def get_commit_diff(commit_hash):
    try:
        check_hash_command = ['git', 'cat-file', '-e', commit_hash]
        subprocess.run(check_hash_command, check=True, capture_output=True)

        result = subprocess.run(
            ['git', 'show', commit_hash, '--patch-with-raw'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if e.cmd[:2] == ['git', 'cat-file']:
             print(f"Error: Commit hash '{commit_hash}' does not exist or is not a valid commit object.")
        else:
            print(f"Error getting diff for commit {commit_hash}: {e}")
            print(f"Stderr output: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: Git does not seem to be installed or in the PATH.")
        return None

def analyze_commit_with_ai(commit_content, model_name="llama3"):
    if not commit_content:
        return "No content to analyze."

    prompt = f"""
    Analyze the following 'git show' output, which includes the original commit message and the diff of the changes.
    Your task is to generate a new, concise, and well-written commit message following Conventional Commits standards,
    based ONLY ON THE CHANGES (the diff). Ignore the original commit message present in the input.
    Provide only the commit message itself, without any introductory or concluding phrases.

    The message should start with a type (e.g., feat, fix, docs, style, refactor, test, chore),
    optionally followed by a scope in parentheses, a colon and a space, and then the description.
    Example: feat(api): add user endpoint

    'git show' output:
    {commit_content}

    New suggested commit message (based only on the diff and following Conventional Commits):
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': 'You are an assistant that analyzes Git commits and their diffs to generate improved commit messages according to the Conventional Commits standard.'},
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.5
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error during analysis with Ollama ({model_name}): {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Analyzes commits in a Git repository and suggests new messages using AI with Ollama.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "repository_path",
        type=str,
        help="The local directory path of the Git repository to analyze."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of recent commits to analyze (optional).\n"
             "If not specified, all commits will be analyzed (this may take a long time).\n"
             "This is ignored if --hash is specified."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Name of the Ollama model to use (e.g., llama3, mistral, codellama, tavernari/git-commit-message).\n"
             "Default: llama3. Ensure it has been pulled with 'ollama pull <model_name>'."
    )
    parser.add_argument(
        "--hash",
        type=str,
        default=None,
        help="The specific commit hash to analyze (optional).\n"
             "If specified, only this commit will be analyzed."
    )

    parser.epilog = """
Examples:
  # Analyze all commits in './my-repo' using the default 'llama3' model
  python IAcommit.py ./my-repo

  # Analyze the last 5 commits in '/path/to/another/repo' using the 'mistral' model
  python IAcommit.py /path/to/another/repo --limit 5 --model mistral

  # Analyze a specific commit 'a1b2c3d' in './my-repo'
  python IAcommit.py ./my-repo --hash a1b2c3d

  # Analyze a specific commit using 'codellama' model
  python IAcommit.py "./my project repo" --hash a1b2c3d --model codellama
"""

    args = parser.parse_args()

    repo_path = os.path.abspath(args.repository_path)
    num_limit = args.limit
    ollama_model = args.model
    specific_hash = args.hash

    if not check_if_git_repo(repo_path):
        sys.exit(1)

    try:
        original_cwd = os.getcwd()
        os.chdir(repo_path)
        print(f"Changed working directory to: {os.getcwd()}")
    except OSError as e:
        print(f"Error changing directory to '{repo_path}': {e}")
        sys.exit(1)

    print(f"\nStarting commit analysis in repository: {repo_path}")
    print(f"Using Ollama model: {ollama_model}")

    commit_hashes_to_analyze = []

    if specific_hash:
        print(f"Analyzing specific commit: {specific_hash}")
        try:
            subprocess.run(['git', 'cat-file', '-e', specific_hash], check=True, capture_output=True, cwd=repo_path)
            commit_hashes_to_analyze = [specific_hash]
        except subprocess.CalledProcessError:
            print(f"Error: Commit hash '{specific_hash}' does not exist or is not a valid commit object in this repository.")
            commit_hashes_to_analyze = []
        except FileNotFoundError:
            print("Error: Git does not seem to be installed or in the PATH.")
            sys.exit(1)
        if num_limit:
            print("Note: --limit is ignored when a specific --hash is provided.")
    else:
        if num_limit:
            print(f"At most, the last {num_limit} commits will be analyzed.")
        else:
            print("WARNING: All commits in the repository will be analyzed (this may take a long time).")
        commit_hashes_to_analyze = get_commit_hashes(limit=num_limit)

    if not commit_hashes_to_analyze:
        if not specific_hash:
             print("No commits found to analyze or error retrieving commits.")
    else:
        total_commits = len(commit_hashes_to_analyze)
        print(f"Found {total_commits} commit(s) to analyze.")

        for index, commit_hash in enumerate(commit_hashes_to_analyze if specific_hash else reversed(commit_hashes_to_analyze)):
            print(f"\n--- Analyzing Commit {index + 1}/{total_commits}: {commit_hash} ---")
            commit_content = get_commit_diff(commit_hash)

            if commit_content:
                print(f"Sending diff for commit {commit_hash} to Ollama (model: {ollama_model})...")
                ai_analysis_result = analyze_commit_with_ai(commit_content, model_name=ollama_model)
                print(f"\nAI suggestion for {commit_hash}:")
                print("---------------------------------- SUGGESTED MESSAGE ----------------------------------")
                print(ai_analysis_result)
                print("-----------------------------------------------------------------------------------------")
            else:
                if specific_hash:
                    print(f"Could not get content for specified commit {commit_hash}. Analysis aborted for this commit.")
                else:
                    print(f"Could not get content for commit {commit_hash}.")

    try:
        os.chdir(original_cwd)
    except OSError as e:
        print(f"Error restoring original working directory: {e}")

if __name__ == "__main__":
    try:
        ollama.list()
        print("Ollama is accessible. Starting script...")
    except Exception as e:
        print(f"Error: Could not communicate with Ollama. Ensure Ollama is running and configured correctly.")
        print(f"Error details: {e}")
        sys.exit(1)
    main()