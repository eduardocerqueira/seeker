#date: 2022-06-13T17:05:25Z
#url: https://api.github.com/gists/d277e708f774d6ee8743fac6a27076ba
#owner: https://api.github.com/users/jeffparaform

chrome_options = webdriver.ChromeOptions()
chrome_options.set_capability('browserless:token', '93ceec78-8211-4809-b4cd-654dc1067cec')
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")

driver = webdriver.Remote(command_executor='https://chrome.browserless.io/webdriver', options=chrome_options)

driver.get("https://www.example.com")
print(driver.title)
driver.quit()