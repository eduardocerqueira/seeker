#date: 2023-03-23T17:05:30Z
#url: https://api.github.com/gists/49b720c0e2bdd5da024d148bf6da8293
#owner: https://api.github.com/users/ansel2000

from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
import json
import os

# Options are only available since client version 2.3.0
options = UiAutomator2Options().load_capabilities({
    # Set URL of the application under test
    "app" : "DemoApp",
    # Specify device and os_version for testing
    "platformVersion" : "12.0",
    "deviceName" : "Google Pixel 6",
    # Set other BrowserStack capabilities
    'bstack:options' : {
        "projectName" : "First Python project",
        "buildName" : "browserstack-build-1",
        "sessionName" : "BStack first_test",
        # Set your access credentials
        "userName" : "username",
        "accessKey" : "accesskey",
        "networkLogs" : "true"
    }
})
# Initialize the remote Webdriver using BrowserStack remote URL
# and options defined above
driver = webdriver.Remote("http://hub.browserstack.com/wd/hub", options=options)
# Test case for the BrowserStack sample Android app.
# If you have uploaded your app, update the test case here.
search_element = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable((AppiumBy.ACCESSIBILITY_ID, "Search Wikipedia"))
)
search_element.click()
search_input = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable(
        (AppiumBy.ID, "org.wikipedia.alpha:id/search_src_text"))
)
search_input.send_keys("BrowserStack")
time.sleep(5)
search_results = driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.TextView")
assert (len(search_results) > 0)
response = driver.execute_script('browserstack_executor: {"action": "getSessionDetails"}')
response= json.loads(response)
driver.quit()

time.sleep(60)
print(str(response['build_hashed_id']), str(response['hashed_id']))
logs = requests.get(
    'https://api-cloud.browserstack.com/app-automate/builds/' + str(response['build_hashed_id']) + '/sessions/' + str(response['hashed_id']) + '/networklogs',
    auth=('username', 'accesskey'),
    headers = {"Content-type": "application/json"},
)
print(logs.content.decode())

json_object = json.dumps(logs.content.decode(), indent=4)

filename = "test1.json"
with open(filename, "w") as outfile:
    outfile.write(json_object)