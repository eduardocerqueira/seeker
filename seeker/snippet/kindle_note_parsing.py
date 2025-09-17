#date: 2025-09-17T17:00:31Z
#url: https://api.github.com/gists/731e8d85aad0b17cada8121679f54ab3
#owner: https://api.github.com/users/calyptis

from bs4 import BeautifulSoup
import re
import pandas as pd

# HTML File
FILEPATH = "..."

soup = BeautifulSoup(open(FILEPATH), "html.parser")

notes = soup.find_all("div", class_="noteText")
notes = [i.text.replace("\n", "").strip() for i in notes]

references = soup.find_all("div", class_="noteHeading")
references = [re.findall("\d+", i.text) for i in references]

df = pd.DataFrame({
    "note": notes, 
    "page": [i[0] for i in references],
    "location": [i[1] for i in references]
}).sort_values(by=["page", "location"])