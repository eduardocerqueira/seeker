#date: 2021-09-21T17:11:19Z
#url: https://api.github.com/gists/5d66414e8fe70961fa8032d4cce08dbc
#owner: https://api.github.com/users/marcosan93

# Adding adblocker extension
options = Options()

options.add_extension(
    "/Users/marcosantos/Downloads/extension_1_37_2_0.crx"
)

# Opening the browser
driver = webdriver.Chrome(
    executable_path="/Users/marcosantos/Downloads/chromedriver",
    options=options
)

# Designating which site to open to
driver.get("https://www.google.com")

time.sleep(1)
# Typing into search and hitting enter
driver.find_element_by_xpath(
    "//input[@class='gLFyf gsfi']").send_keys("yahoo finance", 
                                              Keys.ENTER)

time.sleep(1)
# Clicking on the desired result
driver.find_element_by_xpath("//div[@class='tF2Cxc']").click()

time.sleep(1)
# Searching a Ticker
driver.find_element_by_xpath("//input[@id='yfin-usr-qry']").send_keys("AMC", Keys.ENTER)

time.sleep(1)
# Clicking Historical Data
driver.find_element_by_xpath("//li[@data-test='HISTORICAL_DATA']").click()

time.sleep(1)
# Getting historical date range
driver.find_element_by_xpath(
    "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/div[1]/div/div/div[1]/span"
).click()

time.sleep(1)
# Getting max historical range
driver.find_element_by_xpath(
    "//*[@id='dropdown-menu']/div/ul[2]/li[4]/button"
).click()

time.sleep(1)
# Applying the changes
driver.find_element_by_xpath(
    "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/button/span"
).click()

time.sleep(1)
# Downloading the file
driver.find_element_by_xpath(
    "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[2]/span[2]/a/span"
).click()