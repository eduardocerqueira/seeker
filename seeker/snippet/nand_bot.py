#date: 2024-03-14T16:57:20Z
#url: https://api.github.com/gists/ab8fa597755f9571aabc4df318f58751
#owner: https://api.github.com/users/zhixiangteoh

#!/usr/bin/env python3

# import requests
# from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By


def get_survey_code() -> str:
    # e.g., 2124-1712-1162-0332-0812-07 or 2124171211620332081207
    survey_code = input('Enter the 22-digit survey code: ')
    survey_code = survey_code.replace('-', '')
    return survey_code


def login_survey_with_code(driver: webdriver, survey_code: str) -> None:
    url = 'https://pandaexpress.com/feedback'
    # fill in code and submit form
    driver.get(url)
    driver.find_element(By.ID, 'CN1').send_keys(survey_code[:4])
    driver.find_element(By.ID, 'CN2').send_keys(survey_code[4:8])
    driver.find_element(By.ID, 'CN3').send_keys(survey_code[8:12])
    driver.find_element(By.ID, 'CN4').send_keys(survey_code[12:16])
    driver.find_element(By.ID, 'CN5').send_keys(survey_code[16:20])
    driver.find_element(By.ID, 'CN6').send_keys(survey_code[20:])
    driver.find_element(By.ID, 'NextButton').click()


def fill_out_survey(driver: webdriver, email: str) -> None:
    # first page
    tds = driver.find_elements(By.TAG_NAME, 'td')
    for td in tds:
        if 'Opt3' in td.get_attribute('class'):
            td.click()
    driver.find_element(By.ID, 'NextButton').click()
    # second page
    tds = driver.find_elements(By.TAG_NAME, 'td')
    for td in tds:
        if 'Opt3' in td.get_attribute('class'):
            td.click()
    driver.find_element(By.ID, 'NextButton').click()
    # third page
    tds = driver.find_elements(By.TAG_NAME, 'td')
    for td in tds:
        if 'Opt3' in td.get_attribute('class'):
            td.click()
    driver.find_element(By.ID, 'NextButton').click()
    # fourth, fifth, sixth, seventh, eigth, ninth pages
    driver.find_element(By.ID, 'NextButton').click()
    driver.find_element(By.ID, 'NextButton').click()
    driver.find_element(By.ID, 'NextButton').click()
    driver.find_element(By.ID, 'NextButton').click()
    driver.find_element(By.ID, 'NextButton').click()
    driver.find_element(By.ID, 'NextButton').click()
    # tenth page
    tds = driver.find_elements(By.TAG_NAME, 'td')
    for td in tds:
        if 'Opt2' in td.get_attribute('class'):  # Opt2 := 'No'
            td.click()
    driver.find_element(By.ID, 'NextButton').click()
    # eleventh page
    tds = driver.find_elements(By.TAG_NAME, 'td')
    for td in tds:
        if 'Opt3' in td.get_attribute('class'):
            td.click()
    driver.find_element(By.ID, 'NextButton').click()
    # twelfth page
    driver.find_element(By.ID, 'NextButton').click()
    # thirteenth page
    spans = driver.find_elements(By.TAG_NAME, 'span')
    for span in spans:
        if 'Opt4' in span.find_element(By.XPATH, '..').get_attribute('class'):
            span.click()
    driver.find_element(By.ID, 'NextButton').click()
    # fourteenth page - email
    inputs = driver.find_elements(By.TAG_NAME, 'input')
    for input in inputs:
        if input.get_attribute('type') != 'hidden':
            input.send_keys(email)
    driver.find_element(By.ID, 'NextButton').click()


def main():
    survey_code = get_survey_code()
    email = input('Enter your email: ')
    driver = webdriver.Firefox()
    login_survey_with_code(driver, survey_code)
    fill_out_survey(driver, email)
    driver.quit()


if __name__ == '__main__':
    main()
