#date: 2024-01-25T16:55:26Z
#url: https://api.github.com/gists/dfee57fa3fcdd4f6d58813554dac9784
#owner: https://api.github.com/users/secemp9

import sys
import platform
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys
import pyperclip
from bs4 import BeautifulSoup


 "**********"d "**********"e "**********"f "**********"  "**********"m "**********"a "**********"i "**********"n "**********"( "**********"l "**********"o "**********"o "**********"m "**********"_ "**********"e "**********"m "**********"a "**********"i "**********"l "**********", "**********"  "**********"l "**********"o "**********"o "**********"m "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"u "**********"r "**********"l "**********"_ "**********"t "**********"o "**********"_ "**********"v "**********"i "**********"s "**********"i "**********"t "**********") "**********": "**********"
    driver = uc.Chrome()
    driver.get("https://www.loom.com/login")

    WebDriverWait(driver, 300).until(
        EC.presence_of_element_located((By.ID, "email"))
    )

    email = driver.find_element(By.ID, "email")
    email.clear()
    email.send_keys(loom_email)

    login_button = driver.find_element(By.ID, "email-signup-button")
    login_button.click()

    original_window = driver.current_window_handle
    all_windows = driver.window_handles
    new_window = [window for window in all_windows if window != original_window][0]
    driver.switch_to.window(new_window)

    WebDriverWait(driver, 300).until(
        EC.element_to_be_clickable((By.ID, "identifierId"))
    ).send_keys(loom_email)

    driver.find_element(By.ID, "identifierNext").click()

    pyperclip.copy(loom_password)
    WebDriverWait(driver, 300).until(
        EC.element_to_be_clickable((By.NAME, "Passwd"))
    ).click()
    password_field = "**********"

    if platform.system() == 'Darwin':  # MacOS
        password_field.send_keys(Keys.COMMAND, 'v')
    else:  # Windows and others
        password_field.send_keys(Keys.CONTROL, 'v')

    driver.find_element(By.ID, "passwordNext").click()
    time.sleep(5)
    driver.switch_to.window(original_window)
    driver.get(url_to_visit)
    time.sleep(5)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    links = soup.find_all('a', class_='video-card_videoCardLink_37D')
    for link in links:
        video_url = link.get('href')
        video_title = link.find('h2').text  # Extracting the text from the h2 tag
        print(f"Title: {video_title}, URL: {video_url}")
    driver.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: "**********"
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
