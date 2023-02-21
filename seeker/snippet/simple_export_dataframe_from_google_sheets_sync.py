#date: 2023-02-21T16:58:47Z
#url: https://api.github.com/gists/5f9de4a0cf3a1cdd3ae17f135cc8618a
#owner: https://api.github.com/users/maintainer64

import requests
import pandas as pd
from io import BytesIO


# https://docs.google.com/spreadsheets/d/1fzRAc7J38i2xLgq3dzJIgnNtcmM3JYbCI7LlmkvIfcc/edit
#                                       |----Идентификатор-книги-(spreadsheet_id)----|
def google_sheet_export_dataframe(
        spreadsheet_id: str,
        sheet_name: str
) -> pd.DataFrame:
    """
    Получить dataframe объект из Google Sheets
    :param spreadsheet_id: Идентификатор книги (получить из ссылки)
    :param sheet_name: Наименование листа
    :return: 
    """
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}" \
          f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    response = requests.get(url=url)
    stream = BytesIO(response.content)
    return pd.read_csv(stream)

pd = google_sheet_export_dataframe(
    "1fzRAc7J38i2xLgq3dzJIgnNtcmM3JYbCI7LlmkvIfcc",
    "Question"
)
print(pd.head())
