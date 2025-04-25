#date: 2025-04-25T17:04:08Z
#url: https://api.github.com/gists/4a9faba9dc86bbe1f390e4dedf111fbe
#owner: https://api.github.com/users/zetaloop

from pathlib import Path as path
import requests

Session = requests.Session()

def download(url: str, dest: path, chunk_size: int = 1024 * 200):
    print(
        f'\r<<< {CYAN}Download{RESET} {ITALIC}"{dest.name}"{RESET}', flush=True, end=""
    )
    mb = 0
    with Session.get(url, stream=True) as resp:
        print(
            f'\r<<< {CYAN}Download{RESET} {ITALIC}"{dest.name}"{RESET} (wait)',
            flush=True,
            end="",
        )
        resp.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / 1024 / 1024
                print(
                    f'\r<<< {CYAN}Download{RESET} {ITALIC}"{dest.name}"{RESET} ({mb:.2f}MB)',
                    flush=True,
                    end="",
                )
    print(
        f'\r<<< {CYAN}Download{RESET} {ITALIC}"{dest.name}"{RESET} ({mb:.2f}MB) {GREEN}OK{RESET}'
    )