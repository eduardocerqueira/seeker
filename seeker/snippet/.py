#date: 2023-07-05T17:08:14Z
#url: https://api.github.com/gists/a47089aed01fcf365c71ecb1b08e01f8
#owner: https://api.github.com/users/shammowla

from selenium import webdriver
from time import sleep

# Open a new Chrome browser window
driver = webdriver.Chrome()

try:
    driver.get('https://www.instagram.com/')
    # TODO: Log in to Instagram using your credentials
    
    # Navigate to your own profile page
    driver.get('https://www.instagram.com/USERNAME/')
    
    # Open the list of users who are following you
    followers_button = driver.find_element_by_css_selector('a[href$="/followers/"]')
    followers_button.click()
    
    # Wait for the list to load
    sleep(2)
    
    # Loop through the list of followers and unfollow them
    user_list = driver.find_element_by_css_selector('div[role="dialog"] ul')
    user_list_items = user_list.find_elements_by_tag_name('li')
    
    for user in user_list_items:
        unfollow_button = user.find_element_by_css_selector('button')
        unfollow_button.click()
        
        # Confirm the unfollow action (optional)
        confirm_button = driver.find_element_by_xpath('//button[text()="Unfollow"]')
        confirm_button.click()
        
        # Wait for the next user to be loaded
        sleep(1)
    
finally:
    driver.quit()
