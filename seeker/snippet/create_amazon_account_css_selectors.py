#date: 2024-02-15T17:04:58Z
#url: https://api.github.com/gists/d92c9342bac993ae9cf0b5d21632ac01
#owner: https://api.github.com/users/ahembd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

driver_path = ChromeDriverManager().install()
service = Service(driver_path)
driver = webdriver.Chrome(service=service)
driver.maximize_window()
# get the path to the ChromeDriver executable

#Amazon Logo
driver.find_element(By.ID, "nav-logo-sprites")
#search
driver.find_element(By.CSS_SELECTOR, "#twotabsearchtextbox")
driver.find_element(By.CSS_SELECTOR, "#ap_customer_name")
driver.find_element(By.CSS_SELECTOR, "#ap_email")
driver.find_element(By.CSS_SELECTOR, "#ap_password")
driver.find_element(By.CSS_SELECTOR, "#ap_password_check")
driver.find_element(By.CSS_SELECTOR, "#continue.a-button-input").click()
# conditions of use
driver.find_element(By.CSS_SELECTOR, "#legalTextRow [href*='condition']")