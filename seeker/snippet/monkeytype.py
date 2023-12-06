#date: 2023-12-06T17:06:40Z
#url: https://api.github.com/gists/67c613b827db4ac34b0c90938e5e2692
#owner: https://api.github.com/users/Shravan-1908

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService

# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Firefox(service=FirefoxService())
driver.get("https://monkeytype.com")
try:
    print("finding cookie popup")
    cookie_popup = driver.find_element(By.ID, "cookiePopup")
    print("clicking reject all")
    rejectbutton = driver.find_element(By.CSS_SELECTOR, "button.rejectAll")
    rejectbutton.click()
except NoSuchElementException:
    ...

print("selecting words -> 25")
words_btn = driver.find_element(By.CSS_SELECTOR, ".mode > div:nth-child(2)")
words_btn.click()
words_btn = driver.find_element(By.CSS_SELECTOR, ".wordCount > div:nth-child(2)")
words_btn.click()

print("writing into input")

words = driver.find_elements(By.CSS_SELECTOR, "div.word")
prompt = driver.find_element(By.CSS_SELECTOR, "input#wordsInput")
for word in words:
    text = "".join([w.text for w in word.find_elements(By.TAG_NAME, "letter")])
    text = text.strip()
    prompt.send_keys(f"{text} ")
