#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

from selenium.webdriver.support.wait import WebDriverWait

# 收信列表显示设置
def test_setup_choose(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    # 收件列表
    sjx1 = driver.find_elements_by_css_selector("div[class='cnt']")
    for i in sjx1:
        if i.text == "收件箱":
            i.click()
    time.sleep(1)
    # 列表显示设置
    driver.find_element_by_css_selector("i[class='iconfont iconset']").click()
    time.sleep(1)
    # 显示/不显示正文摘要
    ch = driver.find_elements_by_css_selector("div[class='switch']")
    ch[1].click()