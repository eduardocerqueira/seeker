#date: 2022-07-18T16:59:46Z
#url: https://api.github.com/gists/fc8b1592836e12c995ec90a91ff18577
#owner: https://api.github.com/users/elaspog

import os
import re
import time
import traceback

from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By

from page_config import *


# MODIFY FROM HERE

DRIVER_TYPE = "undetected_chrome"
#DRIVER_TYPE = "chrome"

IS_HEADLESS_MODE = False
PREVENT_COPY_ERROR_IN_NORMAL_MODE = False


# DO NOT MODIFY FROM HERE

username="the_at_sign_here___@___is_not.good"
password="secret_passwd"

WAIT_TIME = 1
DOWNLOAD_DIR = 'output'


def log(evt, txt):

    return f"{evt:<12} : {txt:<20}"


def wait(sec_to_wait):

    time.sleep(sec_to_wait)
    print(log("waited", f"{sec_to_wait} seconds"))


def find_click_wait(selenium_helper, xpath, sec_to_wait, log_msg = ""):

    element = selenium_helper.driver.find_element(By.XPATH, xpath)
    element.click()
    print(log_msg)
    wait(sec_to_wait)


def find_enter_text(selenium_helper, xpath, text_to_send, sec_to_wait, log_msg = ""):

    element = selenium_helper.driver.find_element(By.XPATH, xpath)
    element.send_keys(text_to_send)
    print(log_msg)
    wait(sec_to_wait)


def navigate_wait(selenium_helper, url, sec_to_wait, log_msg = ""):

    selenium_helper.driver.get(url)
    print(log_msg)
    wait(sec_to_wait)


def shoot_screenshot(selenium_helper, filename, log_msg = ""):

    selenium_helper.driver.get_screenshot_as_file(filename)
    print(log_msg)


class SeleniumHelper():

    def __init__(   self,
                    driver_type,
                    headless=False,
                    download_dir_path=".",
                    window_size=(1600,1000),
    ):
        self.driver = None
        self.driver_type = driver_type
        self.headless = headless
        self.download_dir_path = download_dir_path
        self.window_size = window_size

        if driver_type=="chrome":

            from selenium import webdriver

            self.chrome_options = webdriver.ChromeOptions()
            if self.headless:
                self.chrome_options.add_argument('--headless')
                self.chrome_options.add_argument('--disable-gpu')

            prefs = {'download.default_directory' : os.path.realpath(self.download_dir_path)}
            self.chrome_options.add_experimental_option('prefs', prefs)
            self.driver = webdriver.Chrome(options=self.chrome_options)

        if driver_type=="undetected_chrome":

            import undetected_chromedriver as uc

            if PREVENT_COPY_ERROR_IN_NORMAL_MODE:
                # workaround for problem with pasting text from the clipboard into '@' symbol when using send_keys()
                import pyperclip
                pyperclip.copy('@')

            self.chrome_options = uc.ChromeOptions()
            if self.headless:
                self.chrome_options.add_argument('--headless')
                self.chrome_options.add_argument('--disable-gpu')
            self.driver = uc.Chrome(options=self.chrome_options)

            params = {
                "behavior": "allow",
                "downloadPath": os.path.realpath(self.download_dir_path)
            }
            self.driver.execute_cdp_cmd("Page.setDownloadBehavior", params)

        if self.driver == None:
            raise Exception("Selenium Driver Error.")

        if self.driver:
            self.driver.set_window_size(window_size[0],window_size[1])

    def xxx(self):
        pass




def main():
    try:

        selenium_helper = SeleniumHelper(
            driver_type = DRIVER_TYPE,
            headless = IS_HEADLESS_MODE,
            download_dir_path = DOWNLOAD_DIR,
        )

        navigate_wait(selenium_helper, URL_1, WAIT_TIME, log("navigated to", URL_1))
        find_click_wait(selenium_helper, XPATH__LOGIN_BUTTON, WAIT_TIME, log("clicked", "LOGIN button"))
        find_click_wait(selenium_helper, XPATH__SIGN_IN_MENU_BUTTON, WAIT_TIME, log("clicked", "SIGN IN MENU button"))
        find_click_wait(selenium_helper, XPATH__EMAIL_BUTTON, WAIT_TIME, log("selected", "EMAIL option"))
        find_enter_text(selenium_helper, XPATH__EMAIL_INPUT, username, WAIT_TIME, log("entered", "EMAIL"))
        find_enter_text(selenium_helper, XPATH__PASSWORD_INPUT, password, WAIT_TIME, log("entered", "PASSWORD"))
        shoot_screenshot(selenium_helper, "1-error.png", log("screenshot", "2-error.png"))
        find_click_wait(selenium_helper, XPATH__SIGN_IN_BUTTON, WAIT_TIME*2, log("clicked", "SIGN IN button"))
        shoot_screenshot(selenium_helper, "2-error.png", log("screenshot", "2-error.png"))
        navigate_wait(selenium_helper, URL_2, WAIT_TIME*3, log("navigated to", URL_2))


    except Exception as e:
        print(e)
        error_msg = "Error has occured in TradingView Scraper."
        traceback.print_exc()
        print(error_msg)
        exit(-1)


if __name__ == "__main__":
    main()
