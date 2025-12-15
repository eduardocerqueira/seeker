#date: 2025-12-15T16:56:58Z
#url: https://api.github.com/gists/48eb6b8acd00809639cc0bc968a91f91
#owner: https://api.github.com/users/Jyothi-Surla

import json
from playwright.sync_api import sync_playwright, TimeoutError


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Open Amazon
        page.goto("https://www.amazon.de", timeout=60000)

        # Accept cookies if present
        try:
            page.click("input#sp-cc-accept", timeout=5000)
        except TimeoutError:
            pass

        # Search
        page.fill("#twotabsearchtextbox", "Harry Potter Buch")
        page.press("#twotabsearchtextbox", "Enter")

        # Wait for search results container
        page.wait_for_selector("div.s-main-slot", timeout=60000)

        # Get first product link href (robust, no clicking)
        product_link = page.locator(
            "div.s-main-slot a.a-link-normal.s-no-outline"
        ).first.get_attribute("href")

        if not product_link:
            raise Exception("No product link found")

        # Navigate directly to product page
        page.goto(f"https://www.amazon.de{product_link}", timeout=60000)

        # Extract title
        page.wait_for_selector("#productTitle", timeout=60000)
        title = page.locator("#productTitle").inner_text().strip()

        # Extract price
        price = None
        for selector in [
            "span.a-price span.a-offscreen",
            "#priceblock_ourprice",
            "#priceblock_dealprice"
        ]:
            loc = page.locator(selector)
            if loc.count() > 0:
                price = loc.first.inner_text().strip()
                break

        print(json.dumps(
            {"title": title, "price": price},
            ensure_ascii=False
        ))

        browser.close()


if __name__ == "__main__":
    main()
