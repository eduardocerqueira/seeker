#date: 2025-09-23T17:08:24Z
#url: https://api.github.com/gists/57c705a2f4b6501dda0f77978ebc5b05
#owner: https://api.github.com/users/chrdek

import time
import sys
import base64

from seleniumbase import Driver
#import undetected_chromedriver as uc
#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# This will not bypass standard cloudflare captcha
#user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
#uc_chrome_options = uc.ChromeOptions()
# uc_chrome_options.add_argument('--headless=new')
#uc_chrome_options.add_argument("--start-maximized")
#uc_chrome_options.add_argument("user-agent={}".format(user_agent))

# This is ok to cloudflare.
#driver = uc.Chrome(options=uc_chrome_options)
#driver = webdriver.Chrome(options=options)
driver = Driver(uc=True)

driver.get('https://www.urltestdriver.sample.url');
action = ActionChains(driver)

time.sleep(2)

#info section 1 (sample data only)
checkNo = bytes.fromhex('0001').decode('utf-8')
refNoText = driver.find_element(By.CSS_SELECTOR,"#test1")
refNoText.send_keys(checkNo)
submit_check_policy = driver.find_element(By.ID,"submit")
# send keys  (sample data only)
action.send_keys(Keys.DOWN) 
action.send_keys(Keys.DOWN)
action.send_keys(Keys.UP)  
# perform the operation 
action.perform()

time.sleep(5)

submit_check_policy.click()
time.sleep(3)


#Details settings. (sample data only)
number1 = driver.find_element(By.CSS_SELECTOR,'#1st')
number1.send_keys('TEST')
number1 = driver.find_element(By.CSS_SELECTOR,'#2nd')
number1.send_keys('ACCOUNT')
number1 = driver.find_element(By.CSS_SELECTOR,'#Mail')
number1.send_keys('3@test.nf')


#General. details.. (sample data only)
number1 = driver.find_element(By.CSS_SELECTOR,'input[name="txtbox2"]')
number1.send_keys('testnum34')

expPeriod_MM = Select(driver.find_element(By.CSS_SELECTOR,'#DateMM'))
expPeriod_MM.select_by_value('10')
expPeriod_YY = Select(driver.find_element(By.CSS_SELECTOR,'#DateYY'))
expPeriod_YY.select_by_value('2020')
time.sleep(4)

number3 = driver.find_element(By.CSS_SELECTOR,'input[name="txt2"]')
number3.send_keys('testpart4')

# complete section.. (sample data only)
submit_pay = driver.find_element(By.ID,"Submit7")
submit_pay.click()

final_confirm = driver.find_element(By.CSS_SELECTOR, '#Button4')
final_confirm.click()

time.sleep(2) # Close after confirming.. (sample data only)

driver.quit()
sys.exit(0)

# normally ending program..