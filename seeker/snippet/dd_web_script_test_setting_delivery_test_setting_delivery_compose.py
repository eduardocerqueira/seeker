#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-写信设置
from web.script.test_sendmail.test_sendmail_editor import *
from selenium.webdriver.support.ui import Select


def compose_to_name(driver,to_name):
    select = driver.find_element(By.NAME,"default_sender_address")
    options_list = select.find_elements_by_tag_name('option')
    if to_name>0:
        assert  len(options_list)>1,"无别名账号可选，请更换账号测试"
        # 遍历option
        to_name_list=[]
        for option in options_list:
            # 获取下拉框的value和text`
            #print("Value is:%s  Text is:%s" % (option.get_attribute("value"), option.text))
            to_name_list.append(option.get_attribute("value"))
        Select(select).select_by_value(to_name_list[to_name])
        return to_name_list[to_name]
    else:
        pass


def compose_edit_mode(driver,datatype=None):
    ed_html=driver.find_element(By.CSS_SELECTOR,"div:nth-child(1) > label > .radio").get_attribute("class")
    #print(ed_html)
    ed_txt=driver.find_element(By.CSS_SELECTOR,"div:nth-child(2) > label > .radio").get_attribute("class")
    #(ed_txt)
    assert datatype in ["html", "txt"], "参数错误"
    if datatype=="html":
        if 'radio-checked' in ed_html:
            pass
        else:
            driver.find_element(By.CSS_SELECTOR, "div:nth-child(1) > label > .radio").click()
    elif datatype=="txt":
        if 'radio-checked' in ed_txt:
            pass
        else:
            driver.find_element(By.CSS_SELECTOR, "div:nth-child(2) > label > .radio").click()
    else:
        pass
    return datatype



def test_setting_delivery_compose_a(driver, param):
    allure.dynamic.title("收发邮件设置-写信设置")
    param["setting_delivery_compose"]={"compose_fr_name":0,"compose_edit_mode":"html",
                                    "compose_save_sent":False,"compose_smtp_save_sent":False,
                                    'compose_aftersend_saveaddr':False,"compose_displaysender":True}
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.NAME,"default_sender_address")))
    #默认发信账号选择：
    compose_to_name(driver,to_name=int(param["setting_delivery_compose"]["compose_fr_name"])) #0为选择主账号，1为选择别名账号
    #获取选择结果
    param["setting_delivery_compose"]["compose_fr_name_result"]=driver.find_element(By.NAME, "default_sender_address").get_attribute("value")

    #写信时默认编辑模式：单选
    # 多媒体方式(html)#  纯文本方式 txt ['html','txt']
    compose_edit_mode(driver,datatype=param["setting_delivery_compose"]["compose_edit_mode"])
    #写信时：
    #邮件自动保存到[已发送]
    send_checkbox(driver, data=param["setting_delivery_compose"]["compose_save_sent"],ele_data="div:nth-child(1) > label > .checkbox")
    # SMTP发信后保存到 [已发送]
    send_checkbox(driver, data=param["setting_delivery_compose"]["compose_smtp_save_sent"],ele_data="div:nth-child(2) > label > .checkbox")
    # 自动保存收件人到 "个人通讯录"
    send_checkbox(driver, data=param["setting_delivery_compose"]["compose_aftersend_saveaddr"],ele_data="div:nth-child(3) > label > .checkbox")
    #发信人显示我的姓名 :
    send_checkbox(driver, data=param["setting_delivery_compose"]["compose_displaysender"],ele_data="div:nth-child(4) > label > .checkbox")


    # print(to_name)
    # print(edit_mode)
    # print(save_sent)
    # print(smtp_save_sent)
    # print(aftersend_saveaddr)
    # print(displaysender)

    driver.find_element(By.CSS_SELECTOR,".u-btns-top > .u-btn-primary").click()
    result=WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    assert "设置更改成功" in result ,"设置更改失败"
    driver.refresh()




def test_setting_delivery_compose_b(driver,param):
    allure.dynamic.title("收发邮件设置-写信设置完成-检查写信页面")
    setting_delivery_compose=(param["setting_delivery_compose"])
    #正文编辑方式与配置一致
    compose_edit_mode=setting_delivery_compose["compose_edit_mode"]
    assert compose_edit_mode==param['send_Read_body_type'],"写信页面正文编辑方式与配置不一致"

    #写信页面是否勾选"邮件自动保存到已发送"
    compose_save_sent=setting_delivery_compose["compose_save_sent"]
    assert compose_save_sent==param["send_Read_save_sent"],"写信页面默认勾选保存到已发送与配置不一致"
    #写信页面默认发件人与配置一致
    compose_fr_name_result=setting_delivery_compose["compose_fr_name_result"]
    assert compose_fr_name_result==param["send_Read_fr_Email"],"写信页面默认发件人与配置不一致"









if __name__ == '__main__':
    pytest.main(['-s','test_setting_delivery_compose.py::test_setting_delivery_compose_a'])
