#date: 2022-08-25T17:16:44Z
#url: https://api.github.com/gists/e1b1da7134532cab29b6f179a8e66627
#owner: https://api.github.com/users/olegkhomenko

# -- Processing video names
BASE_DIR = "/home/okhomenko/sftp/label-bot/templates/Aug10/"
assert BASE_DIR.endswith("/"), "Don't forget the '/'"
if not Path(BASE_DIR).exists():
    raise ValueError(f"{BASE_DIR} doesn't exist")


def get_hash(fpath: Path):
    """/home/username/templates/001.mp4 -> c7b24d9dd0533a378f2fd379d4f1b8a1"""
    return md5(fpath.as_posix().replace(BASE_DIR, "").encode("utf-8")).hexdigest()


VIDEOS = sorted(list(Path(BASE_DIR).rglob("**/*.mp4")))
VIDEO_NAMES = {get_hash(p): p.as_posix() for p in VIDEOS}
VIDEO_NAMES_KEYS = list(VIDEO_NAMES)

json.dump(VIDEO_NAMES, open(BASE_DIR + f"files-{today()}.json", "w"))
# --