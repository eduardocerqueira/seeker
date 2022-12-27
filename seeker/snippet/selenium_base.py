#date: 2022-12-27T16:44:34Z
#url: https://api.github.com/gists/80c543327b4f1822111160db9e01d360
#owner: https://api.github.com/users/Sigmanificient

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

URL = "https://www.google.com"


def main():
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(
        options=options,
        service=Service("/usr/local/bin/geckodriver")
    )

    driver.get(URL)
    driver.implicitly_wait(5)
    driver.close()


if __name__ == '__main__':
    main()
