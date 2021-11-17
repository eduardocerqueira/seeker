#date: 2021-11-17T17:11:18Z
#url: https://api.github.com/gists/efae985af163e3ecf83253936041f205
#owner: https://api.github.com/users/GabrielSGoncalves

import pandas as pd
import requests
import gspread

def read_private_spreadsheets(
    credentials_json: str, sheet_key: str, worksheet: int = 0
) -> pd.DataFrame:
    """Read a private available Google Spreadsheet as a Pandas Dataframe.

    Parameters
    ----------
    credentials_json : str
        Path to JSON file with GCloud Credentials.

    sheet_key : str
        Key associated to the target spreadsheet.

    worksheet : int (default=0)
        Index or name for the target worksheet.

    Returns
    -------
    pd.DataFrame
        Dataframe loaded from the spreadsheet.

    """
    gcloud = gspread.service_account(filename=credentials_json)
    sheet = gcloud.open_by_key(sheet_key)
    worksheet = sheet.get_worksheet(worksheet)
    list_rows_worksheet = worksheet.get_all_values()
    return pd.DataFrame(
        list_rows_worksheet[1:], columns=list_rows_worksheet[0]
    )