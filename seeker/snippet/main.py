#date: 2021-12-03T17:08:17Z
#url: https://api.github.com/gists/bde6495826835d6154d01e69545c1726
#owner: https://api.github.com/users/k8scat

import random
import string
import time

from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


def wait(d: WebDriver):
    d.implicitly_wait(30)


def main():
    driver = webdriver.Chrome()

    try:
        driver.get('http://192.168.8.1/html/index.html')
        wait(driver)

        # login hilink
        password_input: WebElement = driver.find_element(by=By.ID, value='login_password')
        password_input.send_keys('Holdon@7868')
        login_btn: WebElement = driver.find_element(by=By.ID, value='login_btn')
        login_btn.click()
        time.sleep(5)

        driver.get('http://192.168.8.1/html/content.html#wifieasy')
        wait(driver)

        # update wifi password
        wifi_password_input: WebElement = driver.find_element(by=By.ID, value='wifi_2g_wpa_key')
        wifi_password_input.clear()
        wifi_random_password = ''.join(random.sample(string.digits + string.ascii_letters, 20))
        print(wifi_random_password)
        wifi_password_input.send_keys(wifi_random_password)
        wifisettings_save_btn: WebElement = driver.find_element(by=By.ID, value='wifi_btn_save')
        wifisettings_save_btn.click()

        time.sleep(10)
    finally:
        print('Closing driver')
        driver.quit()


if __name__ == '__main__':
    main()
