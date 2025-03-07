#date: 2025-03-07T17:01:50Z
#url: https://api.github.com/gists/25628c0ce81bdc090be068bf1125ca02
#owner: https://api.github.com/users/huuthan00

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def run(driver):
    driver.get("https://www.example.com")
    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "some_element_id"))
        )
        element.send_keys("Hello, world!")
        button = driver.find_element(By.ID, "submit_button")
        button.click()
    except Exception as e:
        print(f"Error in script: {e}")