#date: 2025-10-30T17:01:58Z
#url: https://api.github.com/gists/3ff0b289e98c0bcdc4de2a5d1490c9ee
#owner: https://api.github.com/users/mjpanula

from playwright.sync_api import Playwright, sync_playwright, expect
import pandas as pd  # Import pandas

def run(playwright: Playwright) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://rpachallenge.com/")
    with page.expect_download() as download_info:
        with page.expect_popup() as page1_info:
            page.get_by_role("link", name="Download Excel cloud_download").click()
        page1 = page1_info.value
    download = download_info.value
    page1.close()
    
    # Save the file to a specific location
    download_path = r"c:\Users\k5001199\Documents\playwright-test\challenge.xlsx"
    download.save_as(download_path)
    print(f"File saved to: {download_path}")

    # Load the data from the Excel file
    df = pd.read_excel(download_path)  # Load the Excel file into a DataFrame
    
    ##############
    # Toteuta alla olevaan väliin toistorakenne (looppi, while tai for)
    # joka käyttää excel-tiedostosta tulevia arvoja
    # ja syöttää ne rpachallenge-palveluun yksitellen
    # tällä pitäisi haaste ratketa

    # Täällä on tietoa eri locator-vaihtoehdoista:
    # https://playwright.dev/python/docs/locators

    # selaimen developer tools aukeaa F12 -napilla
    # se on hyödyllinen apuväline sopivien locaattorien paikallistamiseen

    # Huom! muista asentaa tarvittavat kirjastot
    ############################# TÄSTÄ ALKAA ###############################

    # Tulosta yksittäisiä soluja (cells) DataFramesta
    print("First row, first column:", df.iloc[0, 0])  # Accessing the first cell
    print("First row, second column:", df.iloc[0, 1])  # Accessing the second cell in the first row
    print("Second row, first column:", df.iloc[1, 0])  # Accessing the first cell in the second row
    print("Column 'ColumnName':", df['ColumnName'])  # Accessing a specific column by name
    
    page.get_by_role("button", name="Start").click()
    page.locator("input[name=\"czDwL\"]").click()
    page.locator("input[name=\"czDwL\"]").fill("saposti")
    page.locator("#SLlOq").click()
    page.locator("#SLlOq").fill("osoite")
    page.locator("input[name=\"6i2oF\"]").click()
    page.locator("input[name=\"6i2oF\"]").fill("rooli")
    page.locator("#Czio5").click()
    page.locator("#Czio5").fill("firmannimi")
    page.locator("#hEhho").click()
    page.locator("#hEhho").fill("puhelinnumero")
    page.locator("#QZzat").click()
    page.locator("#QZzat").fill("sukunimi")
    page.locator("input[name=\"t084x\"]").click()
    page.locator("input[name=\"t084x\"]").fill("etunimi")
    page.get_by_role("button", name="Submit").click()

    ####################### TÄHÄN ASTI ###########################

    # ---------------------
    context.close()
    browser.close()


with sync_playwright() as playwright:
    run(playwright)