#date: 2021-12-23T16:50:29Z
#url: https://api.github.com/gists/9c1fd9062e0a1570ed46cbf0bfa70939
#owner: https://api.github.com/users/Irfan-Ahmad-byte

from selenium import webdriver
from selenium.webdriver import Chrome, ChromeOptions, Remote, FirefoxOptions
from selenium.webdriver.common.by import By
import requests
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
import time

options = ChromeOptions()

class scraper():

    def __init__(self):
        self.support_url = ''
        self.driver = Remote(
            command_executor='http://localhost:4444/wd/hub',
            options=options
        )
		
     def open_support(self, name:str) -> tuple:
        '''
        this method tries different support URLs for the platform specified. If any of the specified URls doesn't work,
        it'll try Google search to find out.
        params:
                name: str: name of the platform
        returns:
                name
                supportURL
        '''
        time.sleep(2)
        driver_url = f"https://www.google.com/search?q={name}"
        self.driver.get(driver_url)
        WebDriverWait(self.driver, 10, 3)
        try:
            showing_for = self.driver.find_element(By.XPATH, '//*[text()="Showing results for"]/following-sibling::a')
            platform = showing_for.text
            showing_for.click()
        except:
            platform = name
        try:
            self.driver.get(f'https://help.{platform}.com')
            self.base = self.driver.current_url
        except:
            try:
                self.driver.get(f'https://support.{platform}.com')
                self.base = self.driver.current_url
            except:
                try:
                    self.driver.get(f'https://{platform}.com/support')
                    self.base = self.driver.current_url
                except:
                    try:
                        self.driver.get(f'https://{platform}.com/help')
                        self.base = self.driver.current_url
                    except:
                        return 'no support link found'
        self.links_to_fetch_articles.add(str(self.base))
        return platform, str(self.base)