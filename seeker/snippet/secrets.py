#date: 2023-01-09T16:47:12Z
#url: https://api.github.com/gists/becbbab2dcb930c9ea0cd7b4738aab0c
#owner: https://api.github.com/users/kaaloo

import json
import os
from getpass import getpass
from pathlib import Path
from typing import Dict, List, Optional, Union


def load_secrets(secrets: "**********":
    """
    Loads secrets and sets up some env vars and credential files.

    If the `secrets` param is empty, you will be prompted to input a stringified json dict containing your secrets. Otherwise, the secrets will be loaded from the given string or dict.

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

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********"_ "**********"u "**********"s "**********"e "**********"r "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        os.environ["GH_USER"] = "**********"
        os.environ["GH_PAT"] = "**********"
        # provide a custom credential helper to git so that it uses your env vars
        os.system("""git config --global credential.helper '!f() { printf "%s\n" "username= "**********"=$GH_PAT"; };f'""")

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"a "**********"w "**********"s "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"k "**********"e "**********"y "**********"_ "**********"i "**********"d "**********"" "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********": "**********"
        home = Path.home()
        aws_id = "**********"
        aws_key = "**********"
        (home / ".aws/").mkdir(parents=True, exist_ok=True)
        with open(home / ".aws/credentials", "w") as fp:
            fp.write(f"[default]\naws_access_key_id = "**********"= {aws_key}\n")
