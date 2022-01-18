#date: 2022-01-18T17:13:50Z
#url: https://api.github.com/gists/549644da8e9c87ebcbc2d5beb2921060
#owner: https://api.github.com/users/jonathanmishler

import httpx
from bs4 import BeautifulSoup

class SecFiles:
    """Class object to help parse and download the Sec Financial files"""
    
    BASE_URL = "https://www.sec.gov"
    FILES_URL = f"{BASE_URL}/dera/data/financial-statement-data-sets.html"
    
    def get_urls(self) -> List[dict]:
        """Gets the list of URLs for the financial files for each year and quarter on the webpage"""
        r = httpx.get(self.FILES_URL)
        r.raise_for_status()
        page = BeautifulSoup(r.text, "html.parser")
        tbody = page.find("table").find("tbody")
        rows = [row.find_all("td") for row in tbody.find_all("tr")]
        return [self.parse_row(row) for row in rows]