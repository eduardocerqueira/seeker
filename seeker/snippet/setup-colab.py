#date: 2023-07-31T16:50:23Z
#url: https://api.github.com/gists/46b61140d9c17e557e965f3381fdb319
#owner: https://api.github.com/users/aknip

# based on: "**********"://aseifert.com/p/secrets-management-colab/ 
# and this gist: https://gist.github.com/aseifert/3e2089f673a795029d46e625a46fffbb
import json
import os
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional, Union

from IPython.display import display, HTML, JSON


try:
    import google.colab

    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False

DRIVE = (
    Path("/content/drive/MyDrive")
    if IN_COLAB and Path("/content/drive/MyDrive").exists()
    else None
)
COLAB = None
if DRIVE:
    COLAB = DRIVE / "colab" if DRIVE is not None else None
    os.environ["GDRIVE"] = str(DRIVE)
    os.environ["COLAB"] = str(COLAB)


def rdict(d: Dict) -> None:
    display(JSON(d))


def rhtml(html: str) -> None:
    display(HTML(html))


def rimg(img_url: str, width: int = None, height: int = None) -> None:
    display(HTML(f"<img src='{img_url}' />"))


def rheading(text: str, level=1) -> None:
    rhtml(f"<h{level}>{text}</h{level}>")


def load_secrets(
    secrets: "**********"
) -> None:
    """
    Loads secrets and sets up some env vars and credential files.

    If the `secrets` param is empty you will be prompted to input a stringified json dict containing your secrets.

    The following types of credentials are supported:

    GitHub Credentials:
        `github_user`: GitHub Username
        `github_pat`: "**********"

    AWS Credentials:
        `aws_access_key_id`: "**********"
        `aws_secret_access_key`: "**********"
    """

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********"  "**********"a "**********"n "**********"d "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********", "**********"  "**********"s "**********"t "**********"r "**********") "**********": "**********"
        secrets = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        input = getpass("Secrets (JSON string): "**********"
        secrets = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"p "**********"a "**********"t "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        os.environ["GH_USER"] = "**********"
        os.environ["GH_PAT"] = "**********"
        os.environ["GITHUB_TOKEN"] = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"a "**********"w "**********"s "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"k "**********"e "**********"y "**********"_ "**********"i "**********"d "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        home = Path.home()
        aws_id = "**********"
        aws_key = "**********"
        (home / ".aws/").mkdir(parents=True, exist_ok=True)
        with open(home / ".aws/credentials", "w") as fp:
            fp.write(
                f"[default]\naws_access_key_id = "**********"= {aws_key}\n"
            )

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"h "**********"f "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"/ "**********"r "**********"o "**********"o "**********"t "**********"/ "**********". "**********"h "**********"u "**********"g "**********"g "**********"i "**********"n "**********"g "**********"f "**********"a "**********"c "**********"e "**********"/ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"" "**********", "**********"  "**********"" "**********"w "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********"p "**********": "**********"
            fp.write(secrets["hf_token"])

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"w "**********"a "**********"n "**********"d "**********"b "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        netrc = (
            f"machine api.wandb.ai\n\tlogin user\n\tpassword {secrets['wandb_token']}"
        )
        with open("/root/.netrc", "w") as fp:
            fp.write(netrc)


def wget(urls: Union[str, List[str]], silent=True):
    """
    Use wget to download a list of urls.
    """
    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        if "dropbox.com" in url:
            url = url.split("?")[0]
        out_name = url.split("?")[0].rstrip("/").split("/")[-1]
        silent_flag = "-q" if silent else ""
        os.system(f"wget {silent_flag} {url} -O {out_name}")


def git_clone(repos: Union[str, List[str]]):
    user = os.environ.get("GH_USER")
    pat = os.environ.get("GH_PAT")

    if user:
        assert (
            pat
        ), "please call load_secrets() first and provide both `github_user` and `github_pat`"

    if isinstance(repos, str):
        repos = [repos]

    for repo_name in repos:
        if ".git" not in repo_name:
            if "@" in repo_name:
                repo_name = repo_name.replace("@", ".git@")
            else:
                repo_name += ".git"

        if user:
            os.system(f"git clone https://{user}:{pat}@github.com/{repo_name}")
        else:
            os.system(f"git clone https://github.com/{repo_name}")


def pip_install(*, packages: Union[str, List[str]], silent=True, force_install=False):
    user = os.environ.get("GH_USER")

    if isinstance(packages, str):
        packages = [packages]

    def _get_package_name(full_descriptor):
        return (
            full_descriptor
            if "/" not in full_descriptor
            else full_descriptor.split("/")[-1].split("@")[0]
        )

    package_names = " ".join(
        [
            (pkg.split("/")[-1].split("@")[0] + "/") if "/" in pkg else pkg
            for pkg in packages
        ]
    )

    # install github repos
    git_clone(user=user, repos=[pkg for pkg in packages if "/" in pkg])

    # install packages
    silent_flag = "-qqq" if silent else ""
    os.system(f"pip install {silent_flag} {package_names}")


def report_gpu():
    # import GPUtil
    # for i, gpu in enumerate(GPUtil.getGPUs()):
    #     print(f'[GPU {i}] Mem Free: {gpu.memoryFree/1024:.2f}/{gpu.memoryTotal/1024:.2f} GB | Utilization: {int(gpu.memoryUtil*100)}%')
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(
            f"[{name}] RAM: {torch.cuda.memory_allocated(0)/1024**3:.2f}/{torch.cuda.memory_reserved(0)/1024**3:.2f} GB used"
        )


if IN_COLAB:
    load_secrets()
    report_gpu()