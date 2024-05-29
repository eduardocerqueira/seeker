#date: 2024-05-29T17:08:36Z
#url: https://api.github.com/gists/1bc38656af62cfbca57912ed5e076f5c
#owner: https://api.github.com/users/jarof13

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from bs4 import BeautifulSoup
import pandas 
import time

def webScrapper_Player_Stats(path: str) -> pandas.DataFrame:
    """This function scrapes player stats general traditional from a webpage and returns a DataFrame."""
    ### https://www.nba.com/stats/players/traditional?PlayerPosition=C&sort=PTS&dir=-1&PerMode=Totals&SeasonType=Regular+Season
    # Web Driver
    driver = webdriver.Edge('path/to/msedgedriver')
    driver.get(path)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    data_list = []

    # Find the dropdown for selecting seasons
    select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))

    # Iterate through the dropdown options for Season
    for season_option_index in range(len(select_season_dropdown.options)):
        # Select a season option
        select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))  # Find the dropdown again
        select_season_dropdown.select_by_index(season_option_index)

        # Wait for page to load
        time.sleep(5)

        # Find the dropdown for pagination
        dropdowns = driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")
        
        # Check if the dropdowns are found and index 25 exists
        if len(dropdowns) > 25:
            select_page_dropdown = Select(dropdowns[25])
        else:
            continue

        # Get the number of pages for the current season
        num_pages = len(select_page_dropdown.options)

        # Iterate through the dropdown options for Pagination
        for page_option_index in range(1, num_pages):  # Start from 1 to skip the header row
            try:
                # Select a pagination option
                select_page_dropdown = Select(driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")[25])  # Find the dropdown again
                select_page_dropdown.select_by_index(page_option_index)

                # Wait for page to load
                time.sleep(5)

                # Use BeautifulSoup to parse the page
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Find the table in the HTML
                table = soup.find("table", {"class": "Crom_table__p1iZz"})

                # Check if the table was found
                if table is not None:
                    # Process and extract the information from the table
                    rows = table.find_all("tr")

                    for row in rows[1:]:  # Start from 1 to skip the header row
                        try:
                            cols = row.find_all("td")
                            if len(cols) >= 29:
                                data = {
                                    "Season": select_season_dropdown.first_selected_option.text.strip(),
                                    "Player": cols[1].text.strip(),
                                    "Team"  : cols[2].text.strip(),
                                    "Age"   : cols[3].text.strip(),
                                    "GP"    : cols[4].text.strip(),
                                    "W"     : cols[5].text.strip(),
                                    "L"     : cols[6].text.strip(),
                                    "MIN"   : cols[7].text.strip(),
                                    "PTS"   : cols[8].text.strip(),
                                    "FGM"   : cols[9].text.strip(),
                                    "FGA"   : cols[10].text.strip(),
                                    "FG%"   : cols[11].text.strip(),
                                    "3PM"   : cols[12].text.strip(),
                                    "3PA"   : cols[13].text.strip(),
                                    "3P%"   : cols[14].text.strip(),
                                    "FTM"   : cols[15].text.strip(),
                                    "FTA"   : cols[16].text.strip(),
                                    "FT%"   : cols[17].text.strip(),
                                    "OREB"  : cols[18].text.strip(),
                                    "DREB"  : cols[19].text.strip(),
                                    "REB"   : cols[20].text.strip(),
                                    "AST"   : cols[21].text.strip(),
                                    "TOV"   : cols[22].text.strip(),
                                    "STL"   : cols[23].text.strip(),
                                    "BLK"   : cols[24].text.strip(),
                                    "PF"    : cols[25].text.strip(),
                                    "FP"    : cols[26].text.strip(),
                                    "DD2"   : cols[27].text.strip(),
                                    "DD3"   : cols[28].text.strip(),
                                    "Plus_Minum": cols[29].text.strip()
                                }
                                data_list.append(data)
                        except StaleElementReferenceException:
                            # Handle the exception and refetch the elements if necessary
                            pass
            except NoSuchElementException:
                print(f"No such element found for page index: {page_option_index}")

    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()

    return df

def webScrapper_Player_Stats_Misc(path: str) -> pandas.DataFrame:
    """This function scrapes player stats MISCELLANEOUS from a webpage and returns a DataFrame."""
    ### https://www.nba.com/stats/players/misc?SeasonType=Regular+Season&PerMode=Totals
    # Web Driver
    driver = webdriver.Edge('path/to/msedgedriver')
    driver.get(path)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    data_list = []

    # Find the dropdown for selecting seasons
    select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))

    # Iterate through the dropdown options for Season
    for season_option_index in range(len(select_season_dropdown.options)):
        # Select a season option
        select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))  # Find the dropdown again
        select_season_dropdown.select_by_index(season_option_index)

        # Wait for page to load
        time.sleep(5)

        # Find the dropdown for pagination
        dropdowns = driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")
        
        # Check if the dropdowns are found and index 25 exists
        if len(dropdowns) > 25:
            select_page_dropdown = Select(dropdowns[25])
        else:
            continue

        # Get the number of pages for the current season
        num_pages = len(select_page_dropdown.options)

        # Iterate through the dropdown options for Pagination
        for page_option_index in range(1, num_pages):  # Start from 1 to skip the header row
            try:
                # Select a pagination option
                select_page_dropdown = Select(driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")[25])  # Find the dropdown again
                select_page_dropdown.select_by_index(page_option_index)

                # Wait for page to load
                time.sleep(5)

                # Use BeautifulSoup to parse the page
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Find the table in the HTML
                table = soup.find("table", {"class": "Crom_table__p1iZz"})

                # Check if the table was found
                if table is not None:
                    # Process and extract the information from the table
                    rows = table.find_all("tr")

                    for row in rows[1:]:  # Start from 1 to skip the header row
                        try:
                            cols = row.find_all("td")
                            if len(cols) >= 14:
                                data = {
                                    "Season"    : select_season_dropdown.first_selected_option.text.strip(),
                                    "Player"    : cols[1].text.strip(),
                                    "PTS OFF TO": cols[8].text.strip(),
                                    "2ND PTS"   : cols[9].text.strip(),
                                    "FBPS"      : cols[10].text.strip(),
                                    "PITP"      : cols[11].text.strip(),
                                    "OPP POT"   : cols[12].text.strip(),
                                    "OPP 2ND"   : cols[13].text.strip(),
                                    "OPP FBP"   : cols[14].text.strip(),
                                    "OPP PIT"   : cols[15].text.strip(),
                                    "BLK"       : cols[16].text.strip(),
                                    "BLKA"      : cols[17].text.strip(),
                                    "PF"        : cols[18].text.strip(),
                                    "PFD"       : cols[19].text.strip()
                                }
                                data_list.append(data)
                        except StaleElementReferenceException:
                            # Handle the exception and refetch the elements if necessary
                            pass
            except NoSuchElementException:
                print(f"No such element found for page index: {page_option_index}")

    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()

    return df

def webScrapper_Player_Stats_Advanced(path: str) -> pandas.DataFrame:
    """This function scrapes player stats Advanced from a webpage and returns a DataFrame."""
    ### 
    # Web Driver
    driver = webdriver.Edge('path/to/msedgedriver')
    driver.get(path)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    data_list = []

    # Find the dropdown for selecting seasons
    select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))

    # Iterate through the dropdown options for Season
    for season_option_index in range(len(select_season_dropdown.options)):
        # Select a season option
        select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))  # Find the dropdown again
        select_season_dropdown.select_by_index(season_option_index)

        # Wait for page to load
        time.sleep(5)

        # Find the dropdown for pagination
        dropdowns = driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")
        
        # Check if the dropdowns are found and index 25 exists
        if len(dropdowns) > 25:
            select_page_dropdown = Select(dropdowns[25])
        else:
            continue

        # Get the number of pages for the current season
        num_pages = len(select_page_dropdown.options)

        # Iterate through the dropdown options for Pagination
        for page_option_index in range(1, num_pages):  # Start from 1 to skip the header row
            try:
                # Select a pagination option
                select_page_dropdown = Select(driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")[25])  # Find the dropdown again
                select_page_dropdown.select_by_index(page_option_index)

                # Wait for page to load
                time.sleep(5)

                # Use BeautifulSoup to parse the page
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Find the table in the HTML
                table = soup.find("table", {"class": "Crom_table__p1iZz"})

                # Check if the table was found
                if table is not None:
                    # Process and extract the information from the table
                    rows = table.find_all("tr")

                    for row in rows[1:]:  # Start from 1 to skip the header row
                        try:
                            cols = row.find_all("td")
                            if len(cols) >= 18:
                                data = {
                                    "Season"    : select_season_dropdown.first_selected_option.text.strip(),
                                    "Player"    : cols[1].text.strip(),
                                    "Team"      : cols[2].text.strip(),
                                    "AST%"      : cols[11].text.strip(),
                                    "AST/TO"    : cols[12].text.strip(),
                                    "AST RATIO" : cols[13].text.strip(),
                                    "OREB%"     : cols[14].text.strip(),
                                    "DREB%"     : cols[15].text.strip(),
                                    "REB%"      : cols[16].text.strip(),
                                    "TO RATIO"  : cols[17].text.strip(),
                                    "EFG%"      : cols[18].text.strip(),
                                    "TS%"       : cols[19].text.strip(),
                                    "USG%"      : cols[20].text.strip(),
                                    "PACE"      : cols[21].text.strip(),
                                    "PIE"       : cols[22].text.strip(),
                                    "POSS"      : cols[23].text.strip()
                                }
                                data_list.append(data)
                        except StaleElementReferenceException:
                            # Handle the exception and refetch the elements if necessary
                            pass
            except NoSuchElementException:
                print(f"No such element found for page index: {page_option_index}")

    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()

    return df

def webScrapper_Player_Stats_Scoring(path: str) -> pandas.DataFrame:
    """This function scrapes player stats scoring from a webpage and returns a DataFrame."""
    ### https://www.nba.com/stats/players/scoring?PerMode=Totals&SeasonType=Regular+Season
    # Web Driver
    driver = webdriver.Edge('path/to/msedgedriver')
    driver.get(path)

    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    data_list = []

    # Find the dropdown for selecting seasons
    select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))

    # Iterate through the dropdown options for Season
    for season_option_index in range(len(select_season_dropdown.options)):
        # Select a season option
        select_season_dropdown = Select(driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9"))  # Find the dropdown again
        select_season_dropdown.select_by_index(season_option_index)

        # Wait for page to load
        time.sleep(5)

        # Find the dropdown for pagination
        dropdowns = driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")
        
        # Check if the dropdowns are found and index 25 exists
        if len(dropdowns) > 25:
            select_page_dropdown = Select(dropdowns[25])
        else:
            continue

        # Get the number of pages for the current season
        num_pages = len(select_page_dropdown.options)

        # Iterate through the dropdown options for Pagination
        for page_option_index in range(1, num_pages):  # Start from 1 to skip the header row
            try:
                # Select a pagination option
                select_page_dropdown = Select(driver.find_elements(By.CLASS_NAME, "DropDown_select__4pIg9")[25])  # Find the dropdown again
                select_page_dropdown.select_by_index(page_option_index)

                # Wait for page to load
                time.sleep(5)

                # Use BeautifulSoup to parse the page
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Find the table in the HTML
                table = soup.find("table", {"class": "Crom_table__p1iZz"})

                # Check if the table was found
                if table is not None:
                    # Process and extract the information from the table
                    rows = table.find_all("tr")

                    for row in rows[1:]:  # Start from 1 to skip the header row
                        try:
                            cols = row.find_all("td")
                            if len(cols) >= 14:
                                data = {
                                    "Season"         : select_season_dropdown.first_selected_option.text.strip(),
                                    "Player"         : cols[1].text.strip(),
                                    "FGA2PT%"        : cols[8].text.strip(),
                                    "FGA3PT%"        : cols[9].text.strip(),
                                    "PTS2PT%"        : cols[10].text.strip(),
                                    "PTS2PT MR%"     : cols[11].text.strip(),
                                    "PTS3PT%"        : cols[12].text.strip(),
                                    "PTSFBPS%"       : cols[13].text.strip(),
                                    "PTSFT%"         : cols[14].text.strip(),
                                    "PTSOFFTO%"      : cols[15].text.strip(),
                                    "PTSPITP%"       : cols[16].text.strip(),
                                    "2FGM%AST"       : cols[17].text.strip(),
                                    "2FGM%UAST"      : cols[18].text.strip(),
                                    "3FGM%AST"       : cols[19].text.strip(),
                                    "3FGM%UAST"      : cols[20].text.strip(),
                                    "FGM%AST"        : cols[21].text.strip(),
                                    "FGM%UAST"       : cols[22].text.strip()
                                }
                                data_list.append(data)
                        except StaleElementReferenceException:
                            # Handle the exception and refetch the elements if necessary
                            pass
            except NoSuchElementException:
                print(f"No such element found for page index: {page_option_index}")

    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()

    return df

def webScraper_History_teams_NBA(path:str) -> pandas.DataFrame:

    """This function receives the URL of the nba page which contains the statistical information of the teams and stores the history of all seasons in a Dataframe.
    It is based on the rows of the main table."""
    ### https://www.nba.com/stats/teams/traditional?sort=W_PCT&dir=-1&SeasonType=Regular+Season
    driver      = webdriver.Edge('path/to/chromedriver')
    # URL of the table that you want to obtain the information at a historical level
    driver      .get(path)

    wait        = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    ## Select Filters
    select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
    select      = Select(select_elem)

    data_list   = [] 

    # Iterate through the dropdown options
    for option_index in range(len(select.options)):
        # Select a dropdown option
        select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
        select      = Select(select_elem)
        select      .select_by_index(option_index)

        try:
            # Adjust the wait time as needed
            time.sleep(30)

            # gets the season

            selected_season = select.first_selected_option.text.strip()

            
            # Use BeautifulSoup to parse the page
            soup            = BeautifulSoup(driver.page_source, "html.parser")
            
            # Find the table in the HTML
            table           = soup.find("table", {"class": "Crom_table__p1iZz"})
            
            # Check if the table was found
            if table is not None:
                # Process and extract the information from the table
                rows = table.find_all("tr")
                
                for row in rows[1:]:  # Start from 1 to skip the header row
                    try:
                        cells = row.find_all("td")
                        if len(cells) >= 28:
                            # Rest of the code to collect data
                            data = {
                                "SEASON"    : selected_season,
                                "TEAM"      : cells[1].text.strip(),
                                "GP"        : cells[2].text.strip(),
                                "W"         : cells[3].text.strip(),
                                "L"         : cells[4].text.strip(),
                                "WIN%"      : cells[5].text.strip(),
                                "MIN"       : cells[6].text.strip(),
                                "PTS"       : cells[7].text.strip(),
                                "FGM"       : cells[8].text.strip(),
                                "FGA"       : cells[9].text.strip(),
                                "FG%"       : cells[10].text.strip(),
                                "3PM"       : cells[11].text.strip(),
                                "3PA"       : cells[12].text.strip(),
                                "3P%"       : cells[13].text.strip(),
                                "FTM"       : cells[14].text.strip(),
                                "FTA"       : cells[15].text.strip(),
                                "FT%"       : cells[16].text.strip(),
                                "OREB"      : cells[17].text.strip(),
                                "DREB"      : cells[18].text.strip(),
                                "REB"       : cells[19].text.strip(),
                                "AST"       : cells[20].text.strip(),
                                "TOV"       : cells[21].text.strip(),
                                "STL"       : cells[22].text.strip(),
                                "BLK"       : cells[23].text.strip(),
                                "BLKA"      : cells[24].text.strip(),
                                "PF"        : cells[25].text.strip(),
                                "PFD"       : cells[26].text.strip(),
                                "Plus_Minus": cells[27].text.strip()
                            }
                            
                            data_list.append(data)
                    except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                        pass
        except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                pass
    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()
    return df

def webScraper_History_teams_NBA_Scoring(path:str) -> pandas.DataFrame:

    """This function receives the URL of the nba page which contains the statistical information of the teams and stores the history of all seasons in a Dataframe.
    It is based on the rows of the main table."""
    ### https://www.nba.com/stats/teams/scoring?sort=W&dir=-1&SeasonType=Regular+Season
    
    driver      = webdriver.Edge('path/to/chromedriver')
    # URL of the table that you want to obtain the information at a historical level
    driver      .get(path)

    wait        = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    ## Select Filters
    select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
    select      = Select(select_elem)

    data_list   = [] 

    # Iterate through the dropdown options
    for option_index in range(len(select.options)):
        # Select a dropdown option
        select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
        select      = Select(select_elem)
        select      .select_by_index(option_index)

        try:
            # Adjust the wait time as needed
            time.sleep(45) 

            # gets the season

            selected_season = select.first_selected_option.text.strip()

            
            # Use BeautifulSoup to parse the page
            soup            = BeautifulSoup(driver.page_source, "html.parser")
            
            # Find the table in the HTML
            table           = soup.find("table", {"class": "Crom_table__p1iZz"})
            
            # Check if the table was found
            if table is not None:
                # Process and extract the information from the table
                rows = table.find_all("tr")
                
                for row in rows[1:]:  # Start from 1 to skip the header row
                    try:
                        cells = row.find_all("td")
                        if len(cells) >= 17:
                            # Rest of the code to collect data
                            data = {
                                "SEASON"          : selected_season,
                                "TEAM"            : cells[1].text.strip(),
                                "FGA2PT%"         : cells[6].text.strip(),
                                "FGA3PT%"         : cells[7].text.strip(),
                                "PTS_2PT%"        : cells[8].text.strip(),
                                "PTS_2PT_MR%"     : cells[9].text.strip(),
                                "PTS_3PT%"        : cells[10].text.strip(),
                                "PTS_FBP%"        : cells[11].text.strip(),
                                "PTS_FT%"         : cells[12].text.strip(),
                                "PTS_OFFTO%"      : cells[13].text.strip(),
                                "PTS_PITP%"       : cells[14].text.strip(),
                                "2FGM_AST%"       : cells[15].text.strip(),
                                "2FGM_UAST%"      : cells[16].text.strip(),
                                "3FGM_AST%"       : cells[17].text.strip(),
                                "3FGM_UAST%"      : cells[18].text.strip(),
                                "FGM_AST%"        : cells[19].text.strip(),
                                "FGM_UAST%"       : cells[20].text.strip()
                            }
                            
                            data_list.append(data)
                    except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                        pass
        except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                pass
    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()
    return df

def webScraper_History_teams_NBA_AdvancedStats(path:str) -> pandas.DataFrame:

    """This function receives the URL of the nba page which contains the statistical information of the teams and stores the history of all seasons in a Dataframe.
    It is based on the rows of the main table."""
    ### https://www.nba.com/stats/teams/advanced?sort=W&dir=-1&SeasonType=Regular+Season
    driver      = webdriver.Edge('path/to/chromedriver')
    # URL of the table that you want to obtain the information at a historical level
    driver      .get(path)

    wait        = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DropDown_select__4pIg9")))

    ## Select Filters
    select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
    select      = Select(select_elem)

    data_list   = [] 

    # Iterate through the dropdown options
    for option_index in range(len(select.options)):
        # Select a dropdown option
        select_elem = driver.find_element(By.CLASS_NAME, "DropDown_select__4pIg9")
        select      = Select(select_elem)
        select      .select_by_index(option_index)

        try:
            # Adjust the wait time as needed
            time.sleep(45)  

            # gets the season

            selected_season = select.first_selected_option.text.strip()

            
            # Use BeautifulSoup to parse the page
            soup            = BeautifulSoup(driver.page_source, "html.parser")
            
            # Find the table in the HTML
            table           = soup.find("table", {"class": "Crom_table__p1iZz"})
            
            # Check if the table was found
            if table is not None:
                # Here you can process and extract the information from the table
                rows = table.find_all("tr")
                
                for row in rows[1:]:  # We start from 1 to skip the header row
                    try:
                        cells = row.find_all("td")
                        if len(cells) >= 18:
                            # Rest of the code to collect data
                            data = {
                                "SEASON"    : selected_season,
                                "TEAM"      : cells[1].text.strip(),
                                "MIN"       : cells[5].text.strip(),
                                "OFFRTG"    : cells[6].text.strip(),
                                "DEFRTG"    : cells[7].text.strip(),
                                "NETRG"     : cells[8].text.strip(),
                                "AST%"      : cells[9].text.strip(),
                                "AST/TO"    : cells[10].text.strip(),
                                "ASTRATIO"  : cells[11].text.strip(),
                                "OREB%"     : cells[12].text.strip(),
                                "DREB%"     : cells[13].text.strip(),
                                "REB%"      : cells[14].text.strip(),
                                "TOV%"      : cells[15].text.strip(),
                                "EFG%"      : cells[16].text.strip(),
                                "TS%"       : cells[17].text.strip(),
                                "PACE"      : cells[18].text.strip(),
                                "PIE"       : cells[19].text.strip(),
                                "POSS"      : cells[20].text.strip()
                            }
                            
                            data_list.append(data)
                    except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                        pass
        except StaleElementReferenceException:
                        # Handle the exception and refetch the elements if necessary
                pass
    # Create the DataFrame once all the data has been collected
    df = pandas.DataFrame(data_list)

    driver.quit()
    return df