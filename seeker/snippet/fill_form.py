#date: 2024-06-06T16:39:49Z
#url: https://api.github.com/gists/91dc4e7ec7cbf1de8352b060dbf17bdc
#owner: https://api.github.com/users/onlyoneaman

from playwright.sync_api import sync_playwright

def fill_form(page, label, value):
    field = page.get_by_label(label)
    field.fill(value)

url = "https://forms.gle/fCzgvXeyLfVWP3bU6"

with sync_playwright() as p:
    # Launch the browser
    browser = p.chromium.launch(headless=False, slow_mo=200)

    # Open a new page
    with browser.new_page() as page:
        # Navigate to the URL
        page.goto(url)

        # Fill out the form
        fill_form(page, "Name", "John Doe")
        fill_form(page, "Email", "johndoe@gmail.com")
        fill_form(page, "Message", "Hello, this is a test message")

        # Find and click the submit button
        btn = page.get_by_role("button", name="Submit")
        btn.click()

    # Close the browser
    browser.close()