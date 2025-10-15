#date: 2025-10-15T17:07:05Z
#url: https://api.github.com/gists/1425d3b27677977708bd4fdbcbee3b39
#owner: https://api.github.com/users/null3FF3KT

#!/usr/bin/env python3
"""
Website Screenshot Tool - Capture any website as PNG
"""
import asyncio
import sys
import argparse
from playwright.async_api import async_playwright
import os
from urllib.parse import urlparse

async def url_to_png(url, output_png_path, width=1920, height=1080, full_page=False, wait_time=3000):
    """Convert any URL to PNG using Playwright"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Set viewport size
        await page.set_viewport_size({"width": width, "height": height})
        
        # Navigate to the URL
        print(f"Loading: {url}")
        await page.goto(url, wait_until='networkidle')
        
        # Wait for content to fully load
        await page.wait_for_timeout(wait_time)
        
        # Take screenshot
        if full_page:
            # Full page screenshot
            await page.screenshot(path=output_png_path, type='png', full_page=True)
        else:
            # Viewport screenshot
            await page.screenshot(path=output_png_path, type='png')
        
        # Close browser
        await browser.close()
        
        print(f"Screenshot saved to: {output_png_path}")

def get_output_filename(url, custom_name=None):
    """Generate output filename from URL or use custom name"""
    if custom_name:
        if not custom_name.endswith('.png'):
            custom_name += '.png'
        return custom_name
    
    # Extract domain from URL for filename
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '').replace('.', '_')
    return f"{domain}_screenshot.png"

async def main():
    parser = argparse.ArgumentParser(description='Capture website screenshots as PNG')
    parser.add_argument('url', help='URL to screenshot (include http:// or https://)')
    parser.add_argument('-o', '--output', help='Output PNG filename')
    parser.add_argument('-w', '--width', type=int, default=1920, help='Viewport width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Viewport height (default: 1080)')
    parser.add_argument('-f', '--full-page', action='store_true', help='Capture full page (scroll down)')
    parser.add_argument('-t', '--wait', type=int, default=3000, help='Wait time in ms before screenshot (default: 3000)')
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://")
        sys.exit(1)
    
    # Generate output filename
    output_file = get_output_filename(args.url, args.output)
    
    try:
        await url_to_png(
            url=args.url,
            output_png_path=output_file,
            width=args.width,
            height=args.height,
            full_page=args.full_page,
            wait_time=args.wait
        )
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())