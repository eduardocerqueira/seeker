#date: 2022-07-20T17:14:52Z
#url: https://api.github.com/gists/4722ba6f3ca31dac327af9cf0f832a88
#owner: https://api.github.com/users/ranv1r

from selenium.webdriver.common.by import By
from tauk.tauk_webdriver import Tauk

from test.tauk_test import TaukTest


class WelcomeTest(TaukTest):
    @Tauk.observe(custom_test_name="TaukWelcomeTest", excluded=False)
    def test_ClickPrimaryButton(self):
        self.driver.find_element(
            by=By.CSS_SELECTOR,
            value=".btn-primary"
        ).click()