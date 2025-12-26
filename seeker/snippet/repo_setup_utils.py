#date: 2025-12-26T16:29:08Z
#url: https://api.github.com/gists/f2e180a933792f6e9e03f60a503a9403
#owner: https://api.github.com/users/roguetrainer

"""
Generic Repository Setup Utilities - Self-contained PAT authentication

Single-file utility for GitHub PAT authentication and repository cloning.
Can be used as a public gist or embedded directly in notebooks.

Usage (Option 1 - From Gist):
    import requests
    exec(requests.get('https://gist.githubusercontent.com/YOUR-ID/raw/repo_setup_utils.py').text)
    config = setup_repo_telemetry(
        repo_url='github.com/username/reponame.git',
        module_name='reponame.core'
    )

Usage (Option 2 - Copy into notebook):
    [Copy this file directly into notebook cell]
    config = setup_repo_telemetry(
        repo_url='github.com/username/reponame.git',
        module_name='reponame.core'
    )

Features:
- Environment detection (Colab vs Local)
- GitHub PAT retrieval (Colab Secrets or environment)
- Repository cloning with PAT authentication
- Graceful fallback to standalone mode
- Generic: Works with any repository
"""

import subprocess
import sys
from pathlib import Path


def get_github_pat():
    """Get GitHub PAT from Colab Secrets or environment variable."""
    try:
        from google.colab import userdata
        pat = userdata.get('GITHUB_PAT')
        if pat:
            return pat
    except ImportError:
        pass

    # Fallback to environment variable
    import os
    return os.environ.get('GITHUB_PAT')


def is_colab():
    """Detect if running in Google Colab."""
    try:
        from google.colab import userdata
        return True
    except ImportError:
        return False


def clone_repo(repo_url, dest_dir=None, pat=None):
    """
    Clone a repository with optional PAT authentication.

    Args:
        repo_url: Full GitHub URL (e.g., 'github.com/username/reponame.git')
        dest_dir: Destination directory (default: home directory + repo name)
        pat: "**********"

    Returns:
        dict: Status {'success': bool, 'message': str}
    """
    if dest_dir is None:
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        dest_dir = Path.home() / repo_name

    # If already exists, just add to path
    if dest_dir.exists():
        if str(dest_dir) not in sys.path:
            sys.path.insert(0, str(dest_dir))
        return {'success': True, 'message': f"Crucible already at {dest_dir}"}

    # Try to clone with PAT
    if pat:
        clone_url = f"https://{pat}@{repo_url}"
    else:
        clone_url = f"https://{repo_url}"

    try:
        subprocess.run(
            ["git", "clone", clone_url, str(dest_dir)],
            capture_output=True,
            timeout=60,
            check=True
        )

        if str(dest_dir) not in sys.path:
            sys.path.insert(0, str(dest_dir))

        return {'success': True, 'message': f"Cloned crucible to {dest_dir}"}

    except subprocess.CalledProcessError as e:
        return {'success': False, 'message': f"Clone failed: {e.stderr.decode() if e.stderr else str(e)}"}
    except Exception as e:
        return {'success': False, 'message': f"Unexpected error: {e}"}


def setup_repo_telemetry(repo_url, module_name, verbose=True):
    """
    Setup repository telemetry integration.

    Handles:
    - Environment detection (Colab vs Local)
    - GitHub PAT retrieval (Colab Secrets or env var)
    - Repository cloning (with authentication)
    - Library import verification
    - Graceful fallback to standalone mode

    Args:
        repo_url: GitHub repository URL (e.g., 'github.com/username/reponame.git')
        module_name: Python module to import (e.g., 'crucible.core')
        verbose: Print status messages (default: True)

    Returns:
        dict: Status with keys:
            - 'available' (bool): Whether repository is ready
            - 'environment' (str): 'colab' or 'local'
            - 'repo_url' (str): The repository URL used
    """
    result = {
        'available': False,
        'environment': 'unknown',
        'repo_url': repo_url
    }

    if verbose:
        print("=" * 70)
        print("REPOSITORY TELEMETRY SETUP")
        print("=" * 70)

    # Detect environment
    in_colab = is_colab()
    result['environment'] = 'colab' if in_colab else 'local'

    if verbose:
        print(f"\n📍 Environment: {result['environment'].upper()}")

    # Get GitHub PAT
    pat = get_github_pat()

    if pat:
        if verbose:
            print("✓ GitHub PAT found")
    else:
        if verbose:
            print("⚠ GitHub PAT not configured")
            if in_colab:
                print("  → Click 🔑 (Secrets) and create GITHUB_PAT")
            else:
                print("  → Set: "**********"

    # Try import (may already be installed)
    try:
        exec(f"from {module_name} import *", globals())
        result['available'] = True

        if verbose:
            print(f"✓ {module_name} imported (already installed)")

    except ImportError:
        # Try to clone repository if PAT available
        if pat and verbose:
            print(f"\n⏳ Cloning repository...")

        if pat:
            clone_result = clone_repo(repo_url, pat=pat)
            if clone_result['success'] and verbose:
                print(f"✓ {clone_result['message']}")

            # Try import again
            try:
                exec(f"from {module_name} import *", globals())
                result['available'] = True

                if verbose:
                    print(f"✓ {module_name} now available")

            except ImportError as e:
                if verbose:
                    print(f"⚠ Import still failed: {e}")
                    print("  This is OK - notebook will run in standalone mode")
        else:
            if verbose:
                print(f"⚠ GitHub PAT not available - cannot clone repository")
                print("  Notebook will run in standalone mode (no errors)")

    # Final status
    if verbose:
        print("\n" + "=" * 70)
        if result['available']:
            print("✅ REPOSITORY READY - Advanced telemetry enabled")
        else:
            print("ℹ️  STANDALONE MODE - Telemetry disabled (notebook still works)")
        print("=" * 70 + "\n")

    return result


if __name__ == '__main__':
    # Test when run directly
    # Example: setup_repo_telemetry('github.com/username/reponame.git', 'reponame.core')
    print("Example usage:")
    print("  config = setup_repo_telemetry(")
    print("      repo_url='github.com/username/reponame.git',")
    print("      module_name='reponame.core'")
    print("  )")
    print("\nFunctions available:")
    print("  - is_colab(): Check if running in Colab")
    print("  - get_github_pat(): "**********"
    print("  - clone_repo(repo_url, dest_dir=None, pat=None): Clone a repository")
    print("  - setup_repo_telemetry(repo_url, module_name, verbose=True): Complete setup")
- setup_repo_telemetry(repo_url, module_name, verbose=True): Complete setup")
