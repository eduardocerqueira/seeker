#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

import pytest
import subprocess

@pytest.fixture
def git_info():
    """Fixture that returns the git commit, branch, latest_tag.

    Note if there are uncommitted changes, the commit will have '-dirty' appended.
    """
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        dirty = subprocess.check_output(['git', 'status', '--porcelain']).strip() != b''
        commit = f"{commit}{'-dirty' if dirty else ''}"
    except subprocess.CalledProcessError:
        commit = None
    try:
        latest_tag = subprocess.check_output(['git', 'describe', '--tags', '--abbrev=0']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        latest_tag = None
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    except subprocess.CalledProcessError:
        branch = None
    return {'commit': commit, 'latest_tag': latest_tag, "branch": branch}
  
def test_print_results(module_results_df, git_info):
    """Function that uses pytest-harvest and our custom git fixture that goes at the end of the module to evaluate & save the results."""
    ...
    # add the git information
    module_results_df["git_commit"] = git_info["commit"]
    module_results_df["git_latest_tag"] = git_info["latest_tag"]
    # save results
    module_results_df.to_csv("results.csv")
    ...