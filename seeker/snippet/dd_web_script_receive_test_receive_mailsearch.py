#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

def test_receive_mailsearch(param, driver):
    # 填写搜索关键字
    driver.find_element_by_id("lyfullsearch").send_keys(param["keyword"])
    # driver.find_element_by_id("lyfullsearch").send_keys("李林")
    time.sleep(1)
    # 全文包含“关键字”的邮件
    driver.find_element_by_css_selector("li[class='u-select-item u-select-item-hover']").click()
    time.sleep(1)
    # 发件人包含“关键字”的邮件
    driver.find_element_by_id("lyfullsearch").clear()
    driver.find_element_by_id("lyfullsearch").send_keys("李林")
    time.sleep(1)
    driver.find_element_by_xpath("/html/body/div[1]/div/ul/li[2]").click()
    time.sleep(1)
    # 主题包含“关键字”的
    driver.find_element_by_id("lyfullsearch").clear()
    driver.find_element_by_id("lyfullsearch").send_keys("李林")
    time.sleep(1)
    driver.find_element_by_xpath("/html/body/div[1]/div/ul/li[3]").click()


# 高级搜索
def test_receive_mailhighsearch(param, driver):
    # 点击搜索框
    driver.find_element_by_id("lyfullsearch").click()
    time.sleep(1)
    # 点击高级搜索
    driver.find_element_by_css_selector("li[class='u-select-item u-select-item-hover']").click()
    # 输入关键字、主题、发件人、收件人
    driver.find_element_by_css_selector("input[class='u-input j-keyword']").clear()
    driver.find_element_by_css_selector("input[class='u-input j-keyword']").send_keys("")
    # driver.find_element_by_css_selector("input[class='u-input j-keyword']").send_keys(param['keyword'])
    driver.find_element_by_name("subject").clear()
    driver.find_element_by_name("subject").send_keys("佳人")
    # driver.find_element_by_name("subject").send_keys(param['subject'])
    driver.find_element_by_name("from").clear()
    driver.find_element_by_name("from").send_keys("lilinpeibj@sina.com")
    # driver.find_element_by_name("from").send_keys(param['from'])
    driver.find_element_by_name("to").clear()
    driver.find_element_by_name("to").send_keys("李林蓓")
    # driver.find_element_by_name("to").send_keys(param['to'])
    # 点击“更多”
    driver.find_element_by_css_selector("a[class='grid-col3 conditions-types toggle-more j-toggle']").click()
    time.sleep(1)
    # 点击“确定”
    driver.find_element_by_css_selector("button[class='u-btn u-btn-primary']").click()