#date: 2023-04-25T17:04:10Z
#url: https://api.github.com/gists/31097c2ef3ab2718fa625cd8b1d2e6f9
#owner: https://api.github.com/users/loganbvh

import os

import requests  # pip install requests
from tqdm import tqdm  # pip install tqdm
from tqdm.utils import CallbackIOWrapper

# See: https://gist.github.com/slint/92e4d38eb49dd177f46b02e1fe9761e1
# See: https://gist.github.com/tyhoff/b757e6af83c1fd2b7b83057adf02c139

def get_bucket_url(deposit_id: "**********": str) -> str:
    """Gets the bucket URL for an existing deposit."""
    res = requests.get(
        f"https://zenodo.org/api/deposit/depositions/{deposit_id}", 
        json={},
        params={"access_token": "**********"
    )
    bucket_url = res.json()["links"]["bucket"]
    return bucket_url


def upload_file(
    file_path: str,
    deposit_id: str,
    access_token: "**********"
    progress_bar: bool = True,
) -> None:
    """Uploads a file to an existing Zenodo deposit.
    
    Args:
        file_path: Path to the file to upload.
        deposit_id: Deposit identifier, from https://zenodo.org/deposit/<deposit_id>.
        access_token: "**********":actions" and "deposit:write" scopes.
        progress_bar: Display a progress bar.
    """
    bucket_url = "**********"
    file_size = os.stat(file_path).st_size
    
    with open(file_path, "rb") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=(not progress_bar),
        ) as pbar:
            wrapped_file = CallbackIOWrapper(pbar.update, f, "read")
            requests.put(
                f"{bucket_url}/{os.path.basename(file_path)}",
                data=wrapped_file,
                params=params,
            )d_file,
                params=params,
            )