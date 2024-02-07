#date: 2024-02-07T16:59:42Z
#url: https://api.github.com/gists/52882ed6b93ffd951bb110299b2ae1e9
#owner: https://api.github.com/users/Farzmsh

from selenium.webdriver.common.by import By
from behave import given, when, then
from time import sleep

CIRCLE_LINK = (By.XPATH, "(//a[@data-test='@web/GlobalHeader/UtilityHeader/TargetCircle'])")


@when('open target circle page')
def step_impl(context):
    context.driver.find_element(*CIRCLE_LINK).click()
    sleep(5)