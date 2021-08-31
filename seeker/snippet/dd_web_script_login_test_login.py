#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import allure
from selenium.webdriver.support.wait import WebDriverWait

def test_name_password_login(param,driver):
    allure.dynamic.title("填写用户名密码登录")
    print(param)
    # 等待用户名加载并填写
    WebDriverWait(driver, 20, 0.1).until(
        lambda driver: driver.find_element_by_name('uid'))
    driver.find_element_by_name('uid').clear()
    driver.find_element_by_name('uid').send_keys(param['name'])
    # 填写密码
    driver.find_element_by_id('fakePassword').send_keys(param['password'])
    # 点击登录
    driver.find_element_by_xpath('/html/body/div/div[1]/div/div[5]/div[2]/div[1]/div[2]/div[1]/form/div[3]/button[1]').click()








