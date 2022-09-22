#date: 2022-09-22T17:07:03Z
#url: https://api.github.com/gists/744ac8d8fa83e2aed619249befd8c05c
#owner: https://api.github.com/users/DemonInTheCloset

#!/usr/bin/python
from urllib.request import urlopen
from urllib.error import URLError

URL = str

URLS: dict[str, URL] = {
    "URL": "DYNAMIC_URL"
}


def main() -> None:
    """Script entry point"""
    for domain, url in URLS.items():
        try:
            with urlopen(url) as http:
                with http as page:
                    print(f"{domain:<25}: {page.msg} : {url}")
        except URLError as e:
            print(f"Failed for: {domain}\n{e}")


if __name__ == "__main__":
    main()