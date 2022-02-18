#date: 2022-02-18T16:43:57Z
#url: https://api.github.com/gists/b0940129f772237ce4ac50659013c273
#owner: https://api.github.com/users/joscha0

from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep

chrome_options = Options()


url = 'https://schnelltest.io/schnelltest-cio'
driver = webdriver.Chrome(options=chrome_options)

driver.get(url)
sleep(.3)

# enter information
code = ''
email = ''
postal = ''  # the postal code you think you entered

input_code = driver.find_element(By.ID, "code")
input_code.send_keys(code)
next_button = driver.find_element(
    By.XPATH, "/html/body/div/div/div[2]/div/div/form/div/button")
next_button.click()
sleep(.5)
input_email = driver.find_element(By.ID, "email")
input_email.send_keys(email)
checkbox = driver.find_element(
    By.XPATH, "/html/body/div/div/div[2]/div/div/form/div/div[3]/div")
checkbox.click()

postals = []

# 1 typo
for i in range(5):
    for k in range(10):
        tmp = list(postal)
        tmp[i] = str(k)
        postals.append("".join(tmp))

for i in postals:
    input_postal = driver.find_element(By.ID, "postalCode")
    input_postal.clear()
    input_postal.send_keys(i)
    result_button = driver.find_element(
        By.XPATH, "/html/body/div/div/div[2]/div/div/form/div/button")
    result_button.click()
    sleep(.1)


driver.close()
