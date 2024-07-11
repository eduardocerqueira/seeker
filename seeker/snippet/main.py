#date: 2024-07-11T16:49:14Z
#url: https://api.github.com/gists/6cd650b97c5589b9fe4944d96e5d4bab
#owner: https://api.github.com/users/devniel

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9222")
    default_context = browser.contexts[0]
    page = default_context.pages[0]
    page.goto("https://x.com/devniel/followers")
    page.wait_for_timeout(1000)
    items = page.get_by_test_id("cellInnerDiv").all()
    print(len(items))
    # Get all followers in the page
    users = []
    for item in items:
        links = item.get_by_role("link").all()
        name = links[0].text_content()
        username = links[1].text_content().lstrip("@")
        print(f"{name} | {username}")
        users.append({
            "name": name,
            "username": username
        })
    # Visit all followers profile page
    for user in users:
        print(user)
        page.goto(f"https://x.com/{user["username"]}")
    
