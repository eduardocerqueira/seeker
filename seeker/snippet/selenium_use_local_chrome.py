#date: 2022-07-30T18:50:23Z
#url: https://api.github.com/gists/0efb4ddea9afb92e9191117d2367e522
#owner: https://api.github.com/users/jhcao23

## this example demonstrates how to use existing Chrome app in your Macbook

# pip install selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

chrome_options = webdriver.ChromeOptions()
# make sure to firstly run command to startup chrome debugger then select the profile you want to use
# /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
chrome_options.add_argument('--remote-debugging-port=9222')
chrome_options.add_argument('--disable-gpu')
# download a matching chromedriver from https://chromedriver.chromium.org/downloads
s= Service('./chromedriver')
driver = webdriver.Chrome(service=s, options=chrome_options)

# let's take an example: go to your LinkedIn
# make sure your chrome has already done the login
driver.get("https://www.linkedin.com/")
l = driver.find_element("xpath", '//*[@id="global-nav"]/div/nav/ul/li[2]/a')
l.click()
time.sleep(5)
l = driver.find_element("xpath", '//*[@id="global-nav"]/div/nav/ul/li[5]/a')
l.click()