#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import pytest
from pymemcache.client.base import Client
import allure
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import dill
client = Client(('127.0.0.1', 11211))

def pytest_addoption(parser):
    parser.addoption(
        "--paramTab", action="store", default="param", help="param_tab"
    )

    parser.addoption(
        "--chromePort", action="store", default="9999", help="chromePort"
    )

@pytest.fixture(scope='session')
def param(request):
    paramTab = request.config.getoption("--paramTab")
    print(paramTab)
    param = requsts_util('http://192.168.166.142:8000/memcacheGet/?paramTab=%s'%paramTab,'')
    print(type(param))
    allure.dynamic.feature(param['feature'])
    allure.dynamic.story(param['story'])
    yield param
    print(param)
    time.sleep(1)

@pytest.fixture(scope='session')
def driver(request):
    chromePort = request.config.getoption("--chromePort")
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:%s"%chromePort)
    driver = webdriver.Chrome(options =chrome_options)
    allure.attach(driver.get_screenshot_as_png(), "前截图", allure.attachment_type.PNG)
    yield driver
    allure.attach(driver.get_screenshot_as_png(), "后截图", allure.attachment_type.PNG)
    driver.quit()

@pytest.fixture(scope='session')
def cmdopt(request):
    return request.config.getoption("--paramTab")


def requsts_util(url,param):
    header = {}
    ret = requests.get(url, data=param, headers=header)
    return eval(ret.text)


# if __name__=="__main__":
#     param = requsts_util('http://192.168.166.142:8000/memcacheGet/?paramTab=param_step_id_3', '')
#     print(param)