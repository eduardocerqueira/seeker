#date: 2023-04-17T16:43:57Z
#url: https://api.github.com/gists/2c9b4a3c4aed56f315716730006f02a4
#owner: https://api.github.com/users/rcmadden

import os
from dotenv import load_dotenv
from atlassian import Confluence
from bs4 import BeautifulSoup
import pandas as pd

load_dotenv(dotenv_path="./.env.local")

user = os.environ['CONFLUENCE_USERNAME']
api_key = os.environ['CONFLUENCE_API_KEY']
server = os.environ['BASE_URL']
confluence = "**********"=server, username=user, password=api_key)
page = confluence.get_page_by_title("Space", "Title", expand="body.storage")
body = page["body"]["storage"]["value"]

tables_raw = [[[cell.text for cell in row("th") + row("td")]
               for row in table("tr")] 
               for table in BeautifulSoup(body, features="lxml")("table")]
tables_df = [pd.DataFrame(table) for table in tables_raw]
for table_df in tables_df:
    print(table_df)
# query dataframes
df = pd.DataFrame(tables_df[0])
# summary dataframes
df2 = pd.DataFrame(tables_df[1])])