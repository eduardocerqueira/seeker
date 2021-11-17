#date: 2021-11-17T16:57:32Z
#url: https://api.github.com/gists/57fc7dfbbadbf538e98c4ff298df8de0
#owner: https://api.github.com/users/GabrielSGoncalves

from typing import Union, Dict
from io import StringIO
import json
import pandas as pd
import requests

def read_public_file_from_gdrive(
    file_url: str, file_format: str, **kwargs
) -> Union[pd.DataFrame, Dict]:
    """Generate a Pandas Dataframe from a CSV file in Google Drive.

    Parameters
    ----------
    file_url : str
        Google Drive file URL.

    file_format : str
        Type of file format: 'csv', 'xlsx' or 'json'.

    Returns
    -------
    Union[pd.DataFrame, Dict]
        Dataframe (for 'csv' or 'xlsx') or Dictionary ('json') from Google
        Drive file.

    """
    download_url = (
        "https://drive.google.com/uc?export=download&id="
        + file_url.split("/")[-2]
    )

    request_str_io = StringIO(requests.get(download_url).text)

    if file_format == "csv":
        return pd.read_csv(request_str_io, **kwargs)

    elif file_format == "xlsx":
        return pd.read_excel(request_str_io, **kwargs)

    elif file_format == "json":
        return json.load(request_str_io)