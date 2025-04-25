#date: 2025-04-25T17:06:39Z
#url: https://api.github.com/gists/6dead325884944a64f2c1f41b3be9d87
#owner: https://api.github.com/users/zetaloop

from pathlib import Path as path

def unzip(zip_path: path, dest: path, chunk_size: int = 1024 * 200):
    import zipfile

    zip_path = path(zip_path)
    dest = path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
    total_size = sum(info.file_size for info in infos)
    extracted = 0
    print(
        f'\r<<< {CYAN}Unzip{RESET} {ITALIC}"{zip_path.name}"{RESET}', flush=True, end=""
    )
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in infos:
            target = dest / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(info, "r") as src, open(target, "wb") as out:
                percent = last_percent = ""
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    extracted += len(chunk)
                    last_percent = percent
                    percent = f"{extracted / total_size * 100:.2f}%"

                    if last_percent != percent:
                        filename = info.filename
                        if wclen(filename) < 60:
                            filename = filename + " " * (60 - wclen(filename))
                        elif wclen(filename) > 60:
                            filename = wctail(filename, 60)
                        print(
                            f'\r<<< {CYAN}Unzip{RESET} {ITALIC}"{zip_path.name}"{RESET} {percent} {ITALIC}{filename}{RESET}',
                            flush=True,
                            end="",
                        )
    print(
        f'\r<<< {CYAN}Unzip{RESET} {ITALIC}"{zip_path.name}"{RESET} 100.00% {GREEN}OK{RESET}'
        + " " * 58
    )
    return dest