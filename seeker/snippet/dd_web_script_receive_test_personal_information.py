#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import time

def test_personal_information(param, driver):
    # 回到首页
    driver.find_element_by_xpath("/html/body/section/aside/div[1]/i").click()
    #进入设置页
    driver.find_element_by_xpath("/html/body/section/aside/div[6]/i").click()
    driver.implicitly_wait(10)
    driver.find_element_by_link_text("个人信息").click()
    time.sleep(1)
    driver.find_element_by_xpath("/html/body/section/article/section/div[2]/div[2]/section/article/div[2]/div/section/div/div/section/div[1]/div[1]/div[1]/button[1]").click()
    time.sleep(1)
    # 编辑各项信息
    driver.find_element_by_name("true_name").clear()
    driver.find_element_by_name("true_name").send_keys("李小姐")
    driver.find_element_by_name("nick_name").clear()
    driver.find_element_by_name("nick_name").send_keys("李妹妹")
    driver.find_element_by_name("alt_email").clear()
    driver.find_element_by_name("alt_email").send_keys("lilinpeibj@sina.com")
    driver.find_element_by_name("mobile_number").clear()
    driver.find_element_by_name("mobile_number").send_keys("15810616310")
    driver.find_element_by_name("home_phone").clear()
    driver.find_element_by_name("home_phone").send_keys("85029696")
    driver.find_element_by_name("company_phone").clear()
    driver.find_element_by_name("company_phone").send_keys("89698888")
    driver.find_element_by_name("fax_number").clear()
    driver.find_element_by_name("fax_number").send_keys("66666666")
    driver.find_element_by_name("zipcode").clear()
    driver.find_element_by_name("zipcode").send_keys("065900")
    driver.find_element_by_name("address").clear()
    driver.find_element_by_name("address").send_keys("理工科技大厦")
    driver.find_element_by_name("homepage").clear()
    driver.find_element_by_name("homepage").send_keys("www.baidu.com")
    driver.find_element_by_name("remarks").clear()
    driver.find_element_by_name("remarks").send_keys("这是我的个人邮箱！")
    # 保存
    driver.find_element_by_css_selector("button[class='u-btn u-btn-primary j-form-submit']").click()
    time.sleep(1)
    nname = driver.find_element_by_css_selector("h4[class='nick-name-title']").text

    if nname == "李小姐":
        print("个人信息修改成功！")
    else:
        print("个人信息修改失败！")

def test_personal_signaturen(param, driver):
    driver.find_element_by_css_selector("span[class='signature_position_img']").click()
    time.sleep(2)
    driver.find_element_by_css_selector("button[class='u-btn u-btn-primary j-toggle-panel']").click()

