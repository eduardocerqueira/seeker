#date: 2022-03-15T16:51:28Z
#url: https://api.github.com/gists/917875679d5979405a1851d165104318
#owner: https://api.github.com/users/ccwu0918

# install chromium, its driver, and selenium
!apt update
!apt install chromium-chromedriver
!pip install selenium
# set options to be headless, ..
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
# open it, go to a website, and get results
wd = webdriver.Chrome(options=options)
wd.get("https://www.website.com")
print(wd.page_source)  # results
# divs = wd.find_elements_by_css_selector('div')