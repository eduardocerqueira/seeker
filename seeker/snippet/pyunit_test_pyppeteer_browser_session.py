#date: 2024-09-20T16:57:52Z
#url: https://api.github.com/gists/1e46c343d1cf5910e9faa031171767ed
#owner: https://api.github.com/users/pije76

import asyncio
import unittest
from pyppeteer import connect, launch
from pyppeteer.errors import PageError
from urllib.parse import quote
import json
import os
from os import environ


exec_platform = os.getenv('EXEC_PLATFORM')


# Get username and access key of the LambdaTest Platform
username = environ.get('LT_USERNAME', None)
access_key = "**********"


# Capabilities array with the respective configuration for parallel tests
cloud_capabilities = {
       'browserName': 'Chrome',
       'browserVersion': 'latest',
       'LT:Options': {
           'platform': 'Windows 11',
           'build': '[Build] Launching browser session with Pyppeteer (with unittest)',
           'name': 'Launching browser session with Pyppeteer (with unittest)',
           'user': username,
           'accessKey': "**********"
           'resolution': '1920x1080',
           'network': True,
           'video': True,
           'console': True,
           'headless': False
       }
}


local_capabilities = {
       'browserName': 'Chrome'
}


class LambdaTestAsyncTest(unittest.IsolatedAsyncioTestCase):
   async def asyncSetUp(self):
       if exec_platform == 'cloud':
           capability = quote(json.dumps(cloud_capabilities))
           print('Initializing test:: ', cloud_capabilities['LT:Options']['name'])


           self.browser = await connect(
               browserWSEndpoint=f'wss://cdp.lambdatest.com/puppeteer?capabilities={capability}'
           )
       elif exec_platform == 'local':
           print('Initializing test:: ', local_capabilities['browserName'])
           self.browser = await launch(headless = False, args=['--start-maximized'])


       await asyncio.sleep(1)
       self.page = await self.browser.newPage()


   async def asyncTearDown(self):
       await self.page.close()
       await asyncio.sleep(1)
       await self.browser.close()


   async def test_page_title(self):
       await self.page.goto('https://search.brave.com/')
       title = await self.page.title()
       print('Scenario 1: Page Title ' + title)


       try:
           assert title == 'Private Search Engine - Brave Search', 'Expected page title is incorrect!'
           await self.page.evaluate('_ => {}', f'lambdatest_action: {json.dumps({ "action": "setTestStatus", "arguments": { "status": "passed", "remark": "Title matched" } })}')
       except PageError as e:
           await self.page.evaluate('_ => {}', f'lambdatest_action: {json.dumps({ "action": "setTestStatus", "arguments": { "status": "failed", "remark": str(e) } })}')


   async def test_page_content(self):
       # Navigate to a website to see the effect
       await self.page.goto('https://www.duckduckgo.com')
       element = await self.page.querySelector('[name="q"]')


       await element.click()
       await element.type('LambdaTest')
       await asyncio.gather(
           self.page.keyboard.press('Enter'),
           self.page.waitForNavigation()
       )


       page_title = await self.page.title()
       print('Scenario 2: Page Title ' + page_title)
       return page_title


if __name__ == '__main__':
   unittest.main()
:
   unittest.main()
