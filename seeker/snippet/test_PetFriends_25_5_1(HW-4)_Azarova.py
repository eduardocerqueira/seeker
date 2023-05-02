#date: 2023-05-02T17:02:05Z
#url: https://api.github.com/gists/c6e894d49bcd5b8f02b731280f796c52
#owner: https://api.github.com/users/megaamfibia

import pytest
from selenium.webdriver.chrome import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.fixture(autouse=True)
def test_1():
    pytest.driver = webdriver.Chrome('D:/distrib/chrome/chromedriver.exe')
    pytest.driver.get('http://petfriends.skillfactory.ru/login')
    pytest.driver.find_element(By.NAME, 'email').send_keys('azarova.a@yandex.ru')
    pytest.driver.find_element(By.NAME, 'pass').send_keys('12345678')
    pytest.driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
    assert pytest.driver.find_element(By.TAG_NAME, 'h1').text == "PetFriends"

    yield

    pytest.driver.quit()


def test_table_all_pets():
    assert WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located
                                                  ((By.XPATH, "//a[@href='/my_pets']")))
    assert WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located
                                                  ((By.XPATH, "//a[@href='/all_pets']")))
    assert WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located
                                                  ((By.XPATH, "//button[contains(text(), 'Выйти')]")))


def test_my_pets():
    WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located((By.XPATH,
                                                                             "//button[contains(text(), 'Выйти')]")))
    pytest.driver.find_element(By.XPATH, "//a[@href='/my_pets']").click()
    assert pytest.driver.current_url == 'https://petfriends.skillfactory.ru/my_pets'


def test_elements_of_card():
    pytest.driver.implicitly_wait(10)
    pytest.driver.find_elements(By.CSS_SELECTOR, '.card-deck .card-img-top')
    pytest.driver.find_elements(By.CSS_SELECTOR, '.card-deck .card-title')
    pytest.driver.find_elements(By.CSS_SELECTOR, '.card-deck .card-text')
