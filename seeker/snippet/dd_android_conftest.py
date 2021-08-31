#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import pytest
from pymemcache.client.base import Client
import dill
import allure
import time

client = Client(('localhost', 11211))


@pytest.fixture(scope='session')
def driver():
    param = dill.loads(client.get('param'))
    driver = dill.loads(client.get('driver%s'%param['udid']))
    allure.attach(driver.get_screenshot_as_png(), "前截图", allure.attachment_type.PNG)
    yield driver
    time.sleep(1)
    allure.attach(driver.get_screenshot_as_png(), "后截图", allure.attachment_type.PNG)

@pytest.fixture(scope='session')
def param():
    param = dill.loads(client.get('param'))
    print(param)
    allure.dynamic.feature(param['feature'])
    allure.dynamic.story(param['story'])
    yield param
