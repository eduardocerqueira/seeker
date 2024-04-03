#date: 2024-04-03T16:40:47Z
#url: https://api.github.com/gists/a44307046828e19163f4092cf777aecc
#owner: https://api.github.com/users/eelalzep

"""
Download and save Rekhta.org books as images.

1. Make sure your Chrome browser is closed before running otherwise script will fail.
2. Install selenium (pip install selenium)
3. Install pillow (pip install pillow)
4. Adjust the User directory and User profile values in main()
5. Adjust browser window size in code if requried.
"""

# Original script credit to: Umer Faruk (https://gist.github.com/umerfaruk/89e6cc86425cd9fcfed7e2035ed5c9d0)
# Updated and documented by: Muhammad Yaseen (github.com/muhammadyaseen)
# Element screenshot idea from: 
# https://maxsmolens.org/posts/automating-screenshots-with-selenium/
# https://stackoverflow.com/questions/13832322/how-to-capture-the-screenshot-of-a-specific-element-rather-than-entire-page-usin

from selenium import webdriver
from selenium.webdriver.common.by import By

import time
import os
import argparse

from PIL import Image
from io import BytesIO

# In seconds, depening on your internet speed, you might want to increase it
PAGE_LOAD_WAIT_TIME = 3
REKHTA_PAGE_LIMIT = 5 # Rekhta requires authentication for reading more than 5 pages

# Keep these in 1-place so that future changes are easy
PAGE_RENDER_DIV_ID = "actualRenderingDiv"
NEXT_PAGE_BTN_SELECTOR = ".left.pull-left.ebookprev"

def save_page_image(driver, bookname, page_num):

    # Find the element which renders the book page and take a screenshot of the contents of the element    
    rendered_page_elem = driver.find_element(By.ID, PAGE_RENDER_DIV_ID)
    rendered_page_capture = rendered_page_elem.screenshot_as_png
    rendered_page_capture_as_pil_image = Image.open(BytesIO(rendered_page_capture))

    # Save to disk
    filename = '{0:04d}'.format(page_num)
    rendered_page_capture_as_pil_image.save(f"{bookname}\{filename}.png") 

def click_next_page_button(driver):
    
    next_btn = driver.find_element(By.CSS_SELECTOR, NEXT_PAGE_BTN_SELECTOR)
    
    if not next_btn:
        return False
    else:
        next_btn.click()
        return True

def make_output_folder(bookname):
    
    directory = f"{bookname}"
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_ebook_pages(driver):
    """
    Get total number of pages in this Ebook.
    Could be used for book-keeping (pun intended) purposes e.g. showing a progress bar
    """
    total_pages_elem = driver.find_element(By.CLASS_NAME,"ebookTotalPageCount")
    total_pages_text = total_pages_elem.text

    return int(total_pages_text)

def main(url, bookname):

    make_output_folder(bookname)

    # set-up driver options
    # we need to add these to args otherwise Selenium opens an 'Incognito' like window. 
    # In that window we can't read more than 5-pages. Loading the default profile preserves the login status on Rekhta.org 
    # i.e. if the user if logged in Selenium can successfully read > 5 pages
    options = webdriver.ChromeOptions()
    options.add_argument(r"--user-data-dir=Your User Data directory")
    options.add_argument(r'--profile-directory=YourProfile') #e.g. Default
    
    # Start driver
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(PAGE_LOAD_WAIT_TIME)
    driver.set_window_position(0,0)
    
    # TODO: this should be dependent on client display
    driver.set_window_size(1100,1400)
    driver.get(url)
    
    # wait for page to load
    time.sleep(PAGE_LOAD_WAIT_TIME)

    total_pages = get_ebook_pages(driver)

    if total_pages > REKHTA_PAGE_LIMIT:
        print(f"Your book has more than {REKHTA_PAGE_LIMIT} pages ({total_pages} pages). " \
            "Make sure you're singed in to Rekhta.org otherwise the process will fail.")
    
    page = 1
    while True:

        print(f"Getting page {page} / {total_pages}")
        
        save_page_image(driver, bookname, page)
        moved_next = click_next_page_button(driver)
        
        # wait for page to load after clicking next
        time.sleep(PAGE_LOAD_WAIT_TIME)

        if not moved_next:
            print("Finished saving all pages (or something broke)")
            break 
        page = page + 1

    driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to Download books from rekhta.')

    parser.add_argument('--title', help='Title of book to be imported', required=True)
    parser.add_argument('--url', help='Url of book', required=True)

    args = parser.parse_args()

    main(args.url, args.title)

    input("Press ENTER to exit")

