#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#收发邮件设置-自动转发设置
from web.script.test_sendmail.test_sendmail_editor import *
from selenium.webdriver.support.ui import Select





def forward_switch(driver,datatype=None):
    #自动转发按钮状态
    ele=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff")
    #自动转发按钮
    ele2=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff > .switch")

    if datatype ==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele2.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "开启自动转发成功" in result, "开启自动转发失败"
    elif datatype ==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele2.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "关闭自动转发成功" in result, "自动转发关闭失败"
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")


def forward_insert_email(driver,email):
    ele = driver.find_element(By.CSS_SELECTOR, ".u-switch-onoff")
    if "checkbox-checked" in ele.get_attribute("class"):
        #自动转发地址输入框+添加按钮点击
        driver.find_element(By.CSS_SELECTOR, '.j-new-rule').clear()
        driver.find_element(By.CSS_SELECTOR, '.j-new-rule').send_keys(email)
        driver.find_element(By.CSS_SELECTOR,'.j-button-add').click()



def forward_del_email(driver,email):
    ele=driver.find_elements(By.CSS_SELECTOR,".last-child >.link")
    for i in ele:
        if email==i.get_attribute("data-item-name"):
            i.click()
            WebDriverWait(driver, 5, 0.1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()




def forward_select_emaillist(driver):
    ele = driver.find_elements(By.CSS_SELECTOR, ".last-child >.link")
    ele_list=[]
    for i in ele:
        ele_list.append(i.get_attribute("data-item-name"))
    return ele_list


def forward_active(driver,datatype=None):
    ele1=driver.find_element(By.CSS_SELECTOR,".label > .radio")
    ele2=driver.find_element(By.CSS_SELECTOR,".forwardOpt3 .radio")

    assert datatype in ['任何情況下开启自动转发','当邮箱容量饱和时进行转发'], "参数错误:'展开'or '收起'"
    if datatype=='任何情況下开启自动转发':
        if 'radio-checked' in ele1.get_attribute("class"):
            pass
        else:
            ele1.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "开启自动转发成功" in result, "编辑生效条件-任何情況下开启自动转发，失败"
    elif datatype=='当邮箱容量饱和时进行转发':
        if 'radio-checked' in ele2.get_attribute("class"):
            pass
        else:
            ele2.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "开启自动转发成功" in result, "编辑生效条件-当邮箱容量饱和时进行转发，失败"
    else:
        pass
    return datatype

def forward_keeplocal(driver,datatype=None):
    ele=driver.find_element(By.CSS_SELECTOR,".forwardOpt2 .checkbox")
    if datatype ==True:
        if "checkbox-checked" in ele.get_attribute("class"):
            pass
        else:
            ele.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "开启自动转发成功" in result, "编辑生效条件-同时将邮件保存在本邮箱内，失败"
    elif datatype ==False:
        if "checkbox-checked" in ele.get_attribute("class"):
            ele.click()
            result = WebDriverWait(driver, 5, 0.1).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
            assert "开启自动转发成功" in result, "编辑生效条件-同时将邮件保存在本邮箱内，失败"
        else:
            pass
    else:
        pass
    return ele.get_attribute("class")


def test_setting_delivery_forward_a(driver, param):
    allure.dynamic.title("收发邮件设置-自动转发设置开关与生效条件")
    # param["setting_delivery_forward"]={"forward_switch":True,"forward_active":'当邮箱容量饱和时进行转发',
    #                                    "forward_keeplocal":True}
    """检查是否到达指定页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-switch-onoff > .switch")))

    """页面操作"""
    #自动转发 :True开启， False关闭
    forward_switch(driver,param["setting_delivery_forward"]['forward_switch'])



    '''页面检查'''
    #检查遮罩，关闭后遮罩出现无法进行操作，开启后可进行编辑
    ele = driver.find_element(By.CSS_SELECTOR, ".disable-mask").get_attribute('class')
    switch_data=param["setting_delivery_forward"]['forward_switch']

    if switch_data==True:
        # 生效条件 :['任何情況下开启自动转发','当邮箱容量饱和时进行转发']
        forward_active(driver, param["setting_delivery_forward"]['forward_active'])
        if param["setting_delivery_forward"]['forward_active']=='任何情況下开启自动转发':
            # 任何情況下开启自动转发-是否勾选"同时将邮件保存在本邮箱内"
            forward_keeplocal(driver, param["setting_delivery_forward"]['forward_keeplocal'])
        assert "f-dn" in ele, "预期开启后遮罩消失，现依旧存在"

    elif switch_data==False:
        assert 'f-dn' not in ele ,"预期关闭后遮罩出现，现没有检测到"
        ele1 = driver.find_element(By.CSS_SELECTOR, ".label > .radio")
        ele2 = driver.find_element(By.CSS_SELECTOR, ".forwardOpt3 .radio")
        assert 'radio-checked' not in  ele1.get_attribute("class"),"预期关闭后，生效条件都为未选中"
        assert 'radio-checked' not in ele2.get_attribute("class"),"预期关闭后，生效条件都为未选中"


def test_setting_delivery_forward_add(driver, param):
    # param["forward_insert_email"]="dsdsd1@wo.cn"
    allure.dynamic.title("收发邮件设置-自动转发设置-自动转发地址新增")
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-switch-onoff > .switch")))


    #新增自动转发地址
    forward_insert_email(driver,param["forward_insert_email"])
    try:
        result = WebDriverWait(driver, 5, 0.1).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
    except:
        result = driver.find_element(By.CSS_SELECTOR, ".u-error-alert").text

    assert "自动转发列表更新成功" in result, "新增自动转发地址失败"


    ele_list=forward_select_emaillist(driver)
    assert param["forward_del_email"] in ele_list,'自动转发地址添加失败，列表内没有该新增项'

def test_setting_delivery_forward_addERR(driver, param):
    allure.dynamic.title("收发邮件设置-自动转发设置-自动转发地址格式与重复容错校验")
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-switch-onoff > .switch")))

    """格式容错检查，预期添加失败并提示：邮件地址格式错误"""
    forward_insert_email(driver, "xxxxx")
    result = driver.find_element(By.CSS_SELECTOR, ".u-error-alert").text
    assert "邮件地址格式错误" in result, "容错提示异常"
    ele_list=forward_select_emaillist(driver)
    assert param["forward_del_email"] not in ele_list,'格式错误数据添加成功'

    """重复添加检查，预期添加失败并提示：xxxxx234@wo.cn存在于当前转发名单中，不能重复添加"""
    email="xxxxx2345@wo.cn"
    forward_insert_email(driver, email)
    time.sleep(1)
    forward_insert_email(driver, email)

    result = driver.find_element(By.CSS_SELECTOR, ".u-error-alert").text
    assert "%s存在于当前转发名单中，不能重复添加" % email in result, "容错提示异常"
    ele_list=forward_select_emaillist(driver)
    assert param["forward_del_email"] not in ele_list,'自动转发地址添加失败，列表内没有该新增项'






def test_setting_delivery_forward_del(driver, param):
    param["forward_del_email"]="dsdsd1@wo.cn"
    allure.dynamic.title("收发邮件设置-自动转发设置-自动转发地址删除")
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-switch-onoff > .switch")))

    ele_list=forward_select_emaillist(driver)
    assert param["forward_del_email"] in ele_list,'没有需要删除的对象，请确认参数'

    """删除自动转发地址"""
    forward_del_email(driver, param["forward_del_email"])

    """结果检查"""
    result = WebDriverWait(driver, 5, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text

    assert "自动转发列表更新成功" in result, "删除自动转发地址失败"
    #获取自动转发地址列表内数据

    ele_list2=forward_select_emaillist(driver)
    assert param["forward_del_email"] not in ele_list2,'移除失败'

    ele = driver.find_elements(By.CSS_SELECTOR, ".last-child >.link")
    if len(ele)==0:
        ele2=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff")
        assert "checkbox-checked" not in ele2.get_attribute("class")



def eletyoe(driver):
    try:
        driver.find_element(By.CSS_SELECTOR, ".last-child >.link")
        return True
    except:

        return False


def test_setting_delivery_forward_Alldel(driver):
    allure.dynamic.title("自动转发设置页面-自动转发地址清空（有数据可触发关闭自动转发）")
    """检查是否到达指定页面"""
    WebDriverWait(driver, 10, 0.1).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR,".u-switch-onoff > .switch")))

    """页面操作"""
    #自动转发 :True开启， False关闭

    forward_switch(driver,True)
    ele_list=forward_select_emaillist(driver)

    if len(ele_list)>0:
        for i in range(0,100):
            if  eletyoe(driver)==True:
                print(i)
                try:
                    ele=driver.find_element(By.CSS_SELECTOR, ".last-child >.link")
                    ele.click()
                    WebDriverWait(driver, 5, 0.1).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".u-dialog-btns > .u-btn-primary"))).click()
                    result = WebDriverWait(driver, 5, 0.1).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, ".body"))).text
                except:
                    pass
            else:
                break

        #清空完成，自动关闭转法
        ele2=driver.find_element(By.CSS_SELECTOR,".u-switch-onoff")
        assert "checkbox-checked" not in ele2.get_attribute("class")
    ele_list=forward_select_emaillist(driver)
    assert len(ele_list) == 0,'清理地址列表失败'



if __name__ == '__main__':
    pytest.main(['-s','test_setting_delivery_forward.py::test_setting_delivery_forward_addERR'])
