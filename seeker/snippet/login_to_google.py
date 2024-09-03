#date: 2024-09-03T17:08:18Z
#url: https://api.github.com/gists/7e30a1630f9c2d59d619b4b7e4008e2e
#owner: https://api.github.com/users/samuelvillegas

"""
The most important part is replacing the word `headless` from the user Agent
"""
import selenium.webdriver.chrome.webdriver
import selenium.webdriver.chrome.options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time

email = 'example@mail.com'
password = "**********"

# Setting up Chrome options
driver_opts = selenium.webdriver.chrome.options.Options()
driver_opts.add_argument('--no-sandbox')
driver_opts.add_argument('--disable-dev-shm-usage')

# Disable GPU hardware acceleration
# To prevent gpu related issues
driver_opts.add_argument('--disable-gpu')

driver_opts.add_argument('--window-size=1920,1080')
driver_opts.add_argument('--start-maximized')

# Disable the Blink feature that detects automation
driver_opts.add_argument('--disable-blink-features=AutomationControlled')

# Exclude the 'enable-automation' switch to make automation less detectable
driver_opts.add_experimental_option('excludeSwitches', ['enable-automation'])

# Disable the use of the automation extension
driver_opts.add_experimental_option('useAutomationExtension', False)

# driver_opts.add_argument('--incognito')  # Open the browser in incognito mode

# Run the browser in headless mode
driver_opts.add_argument('--headless')

# Initialize Chrome WebDriver
driver = selenium.webdriver.chrome.webdriver.WebDriver(options=driver_opts)

# Remove 'Headless' from the user agent
# So, websites can't detect that the browser is running in headless mode
current_user_agent = driver.execute_script("return navigator.userAgent;")
driver.execute_cdp_cmd(
    'Network.setUserAgentOverride',
    {
        "userAgent": current_user_agent.replace('Headless', ''),
    },
)

# Prevent websites from detecting that the browser is being controlled by automation tools
driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
    'source': '''
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined
        });
    '''
})

try:
    # Open Google login page
    driver.get('https://accounts.google.com/signin')

    # Enter email or phone
    email_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'identifierId'))
    )
    time.sleep(2)  # Small delay
    email_input.send_keys(email)
    email_input.send_keys(Keys.RETURN)

    # Wait for password input to appear and enter password
    password_input = "**********"
        EC.presence_of_element_located((By.NAME, 'Passwd'))
    )
    time.sleep(2)  # Small delay
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)

    # Wait for successful login (Example: Google home page)
    WebDriverWait(driver, 10).until(
        EC.title_contains('Google'),
    )
    print("Login successful!")

finally:
    # Close the browser
    driver.quit()
ser
    driver.quit()
