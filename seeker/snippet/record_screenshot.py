#date: 2022-02-08T16:58:39Z
#url: https://api.github.com/gists/8c8cd8c6020bd75e101159527e18e8aa
#owner: https://api.github.com/users/theDestI

from selenium import webdriver
from PIL import Image
from io import BytesIO

from selenium.webdriver.chrome.options import Options

from selenium.webdriver import ActionChains

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox') 
chrome_options.add_argument('--disable-dev-shm-usage') # Needed on when running on Ubuntu

chrome = webdriver.Chrome(chrome_options = chrome_options)

URL = 'https://stackoverflow.com/questions/15018372/how-to-take-partial-screenshot-with-selenium-webdriver-in-python'

chrome.get(URL)

# Following code is used to avoid cookie banner being shown on screenshot
# Detect cookie banner
cookie_banner = chrome.find_elements_by_class_name('js-accept-cookies')[0]

# Accept it
ActionChains(chrome).click(cookie_banner).perform()

# Useful feature of SO is that you can actually select responses where certain programming languages were used
# For example, in this case we'll use python
element = chrome.find_elements_by_class_name('python') 

# We'll select first response with .python class because typically StackOverflow thread is sorted by rating
location = element[0].location
size = element[0].size
png = element[0].screenshot_as_png # This is the place where we're capturing screenshot of element

# Dont forget to close connection when it's no longer needed
chrome.quit()

# Using Pillow library to convert raw bytes into .png file
im = Image.open(BytesIO(png))
im.save('screenshot.png')