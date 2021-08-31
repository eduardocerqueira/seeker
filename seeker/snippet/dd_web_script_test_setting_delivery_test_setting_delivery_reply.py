#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-回复设置
from web.script.test_sendmail.test_sendmail_editor import *
from selenium.webdriver.support.ui import Select




def reply_addo(driver,datatype=None):
    ele=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(1) .checkbox")
    if datatype ==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele.click()
    elif datatype ==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")

def reply_value(driver,to_value):
    select = driver.find_element(By.NAME,"replyf")
    options_list = select.find_elements_by_tag_name('option')

    # 遍历option
    to_name_list=[]
    to_value_txt=[]
    for option in options_list:
        # 获取下拉框的value和text`
        #print("Value is:%s  Text is:%s" % (option.get_attribute("value"), option.text))
        to_name_list.append(option.get_attribute("value"))
        to_value_txt.append(option.text)
    Select(select).select_by_value(to_name_list[to_value])
    return to_value_txt[to_value]

def reply_all_mode(driver,datatype=None):
    ed_html=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(3) div:nth-child(1) .radio")
    #print(ed_html)
    ed_txt=driver.find_element(By.CSS_SELECTOR,".u-form-item:nth-child(3) > .u-form-value > div:nth-child(2) .radio")
    #(ed_txt)
    assert datatype in ["转抄送", "保持为收件人"], "参数错误"

    if datatype=="转抄送":
        if 'radio-checked' in ed_html.get_attribute("class"):
            pass
        else:
            ed_html.click()
    elif datatype=="保持为收件人":
        if 'radio-checked' in ed_txt.get_attribute("class"):
            pass
        else:
            ed_txt.click()
    else:
        pass
    return datatype

def test_setting_delivery_reply_a(driver, param):
    allure.dynamic.title("收发邮件设置-回复设置")

    # param["setting_delivery_reply"]={"reply_addo":True,"reply_value":0,
    #                                 "reply_all_mode":"保持为收件人"}
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-form-item:nth-child(1) .checkbox")))
    #包含原信：是与否
    reply_addo(driver, datatype=param["setting_delivery_reply"]["reply_addo"])
    #回复主题前缀：
    #['Re:(推荐选择)', '>', 'Reply:', '回复: (与你所使用的界面语言有关)']
    reply_value(driver, to_value=param["setting_delivery_reply"]["reply_value"])
    #全部回复时原收件人处理：
    #["转抄送","保持为收件人"]
    reply_all_mode(driver,datatype=param["setting_delivery_reply"]["reply_all_mode"])

    #保存
    driver.find_element(By.CSS_SELECTOR,".m-reply-set .u-btn-primary").click()
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "设置更改成功" in result ,"设置更改失败"
    driver.refresh()








if __name__ == '__main__':
    pytest.main(['-s','test_setting_delivery_reply.py::test_setting_delivery_reply_a'])
