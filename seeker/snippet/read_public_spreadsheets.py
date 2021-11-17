#date: 2021-11-17T16:54:13Z
#url: https://api.github.com/gists/61b0f3467b73c048e5d6ff16dd742780
#owner: https://api.github.com/users/GabrielSGoncalves

from io import BytesIO
import requests
import pandas as pd

def read_public_spreadsheets(file_url: str) -> pd.DataFrame:
    """Read a publicly available Google Spreadsheet as a Pandas Dataframe.

    Parameters
    ----------
    file_url : str
        URL adress to the spreadsheet CSV file.

    Returns
    -------
    pd.DataFrame
        Dataframe loaded from the CSV adress.

    """
    response = requests.get(file_url)
    return pd.read_csv(BytesIO(response.content))