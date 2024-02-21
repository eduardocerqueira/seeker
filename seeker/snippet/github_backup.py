#date: 2024-02-21T17:00:53Z
#url: https://api.github.com/gists/8051398f8d7a882d7b5f9164bb755469
#owner: https://api.github.com/users/mvandermeulen

#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from subprocess import call
from argparse import ArgumentParser

# pip install PyGithub
from github import Github
from github.Repository import Repository


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--token",
        help="GitHub access token. Get one at https: "**********"
    )
    arg_parser.add_argument(
        "--compress",
        action="store_true",
        help="Download repo archives rather than clone",
    )
    args = arg_parser.parse_args()

    def clone_repo(repo: Repository) -> None:
        print(f"Cloning {repo.name}...")
        error = call(["git", "clone", "-q", repo.ssh_url])
        if not error and args.compress:
            print(f"Compressing {repo.name}...")
            error = call(["tar", "czf", f"{repo.name}.tgz", repo.name])
            if not error:
                call(["rm", "-rf", repo.name])

    g = "**********"
    executor = ThreadPoolExecutor(max_workers=8)
    executor.map(clone_repo, g.get_user().get_repos())
    try:
        executor.shutdown(wait=True, cancel_futures=False)
    except KeyboardInterrupt:
        print("Shutting down...")
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()_ == "__main__":
    main()