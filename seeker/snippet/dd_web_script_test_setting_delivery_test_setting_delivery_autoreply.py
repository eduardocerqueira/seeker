#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-模板信设置

from web.script.test_sendmail.test_sendmail_editor import *


def autoreply_switch(driver,datatype=None):
    #自动转发按钮状态
    ele=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff")
    #自动转发按钮
    ele2=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff > .switch")
    if datatype==None:
        return send_checkbox_type(ele.get_attribute("class"))
    if datatype ==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele2.click()
    elif datatype ==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele2.click()
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")


def autoreply_time(driver,time_data):
    # filename_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # print(filename_time)
    start_time = driver.find_element(By.NAME, "autorep_condition[begintime][]")
    start_time.clear()
    start_time.send_keys(time_data[0])

    end_time = driver.find_element(By.NAME, "autorep_condition[endtime][]")
    end_time.clear()
    end_time.send_keys(time_data[1])

#通用复选框函数
def autoreply_checkbox(driver,data,ele_data):
    ele=driver.find_element(By.CSS_SELECTOR,ele_data)
    if data==None:
        return send_checkbox_type(ele.get_attribute("class"))
    elif data==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele.click()
    elif data == False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")

def test_setting_delivery_autoreply_a(driver, param):
    allure.dynamic.title("收发邮件设置-假期自动回复设置")
    newwindow(driver,nb=0)
    # param["autoreply_switch"]=True
    # param["autoreply_time"]=("2021-01-01","2023-01-01")
    # param["autoreptext"]="正文"
    # param["autoresp_instation"]=False
    # param["autorep_welcometip"]=False

    """检查是否进入页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR,".u-switch-onoff > .switch")))
    autoreply_switch(driver,True)
    if param["autoreply_switch"]==True:
        #时间范围
        autoreply_time(driver,param["autoreply_time"])

        #正文
        autoreptext=driver.find_element(By.ID,"dest")
        autoreptext.clear()
        autoreptext.send_keys(param["autoreptext"])
        dd=driver.find_element(By.ID,"saveAutoReply").is_enabled()
        print(dd)
        driver.find_element(By.ID,"saveAutoReply").click() #时间输入会导致勾选框被遮挡，现分开保存
        result = WebDriverWait(driver, 5, 0.1).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
        assert "更新自动回复成功" in result, "更新自动回复失败"

        #driver.find_element(By.CSS_SELECTOR, ".m-autoreply").click()
        autoreply_checkbox(driver,param["autoresp_instation"],".u-choice > label:nth-child(1) > .checkbox")
        autoreply_checkbox(driver,param["autorep_welcometip"],"label:nth-child(3) > .checkbox")

        ele=driver.find_element(By.ID,"saveAutoReply")
        if ele.is_enabled()==True:
            print(ele.is_enabled())
            driver.find_element(By.ID, "saveAutoReply").click()  # 时间输入会导致勾选框被遮挡，现分开保存
            time.sleep(1)
            result = WebDriverWait(driver, 10, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "更新自动回复成功" in result, "更新自动回复失败"  
            assert ele.is_enabled() ==False,"保存成功，按钮预期置灰"
        else:
            print(ele.is_enabled())
            assert ele.is_enabled() == False, "保存成功，按钮预期置灰"


def test_setting_delivery_autoreply_b(driver, param):
    allure.dynamic.title("收发邮件设置-假期自动回复设置-回到首页检查开启提示")
    newwindow(driver,nb=0)
    #param["autorep_welcometip"]=True

    WebDriverWait(driver, 5, 0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '.iconhome'))).click()
    driver.refresh()
    WebDriverWait(driver, 5, 0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '.iconhome')))

    if param["autorep_welcometip"]==True:
        ele=driver.find_element(By.CSS_SELECTOR,".panel-autorep").text
        assert "您的自动回复正在生效，会对每一封来信自动回复，您可以"in ele,"未见首页提示我已设置自动回复"
    elif  param["autorep_welcometip"]==False:
        auto_type=True
        try:
            driver.find_element(By.LINK_TEXT, "关闭这个功能").click()
        except:
            auto_type=False
        assert auto_type== False,"预期不勾选首页不可见提示，实际可见"



def test_setting_delivery_autoreply_c(driver, param):
    allure.dynamic.title("收发邮件设置-假期自动回复设置-首页关闭假期自动回复，前置条件：先开启并勾选首页提示")
    newwindow(driver,nb=0)
    #回到首页
    WebDriverWait(driver, 3, 0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '.j-lylogo > img'))).click()

    #在首页点击 "关闭这个功能"
    WebDriverWait(driver, 3, 0.1).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "关闭这个功能"))).click()

    # #捕捉操作提示
    # result = WebDriverWait(driver, 5, 0.1).until(
    #     EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    # assert "已关闭自动回复" in result,"关闭自动回复失败"


    #刷新页面，检查页面不可见该元素
    driver.refresh()
    WebDriverWait(driver, 10, 0.1).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, '.iconhome')))

    ele = True
    try:
        driver.find_element(By.LINK_TEXT, "关闭这个功能").click()
    except:
        ele = False

    assert False == ele, "首页假期自动回复提示关闭失败"




if __name__ == '__main__':
    pytest.main(['-v','test_setting_delivery_autoreply.py::test_setting_delivery_autoreply_b'])
