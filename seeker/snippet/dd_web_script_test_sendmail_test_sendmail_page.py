#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#页面转换按钮点击
import inspect

from web.script.test_sendmail.test_sendmail_editor import *

def page_HF(driver):
    try:
        driver.find_element(By.CSS_SELECTOR, '.toolbar > .u-btns:nth-child(2) > .u-btn:nth-child(1) > .icondown').click()
        time.sleep(1)
        driver.find_element(By.LINK_TEXT, '带附件回复').click()
        return '带附件回复'
    except:
        driver.find_element(By.CSS_SELECTOR, '.toolbar > .u-btns:nth-child(2) > .u-btn:nth-child(1)').click()
        return '回复'

def page_HF_ALL(driver):
    try:
        driver.find_element(By.CSS_SELECTOR, '.toolbar > .u-btns:nth-child(2) > .u-btn:nth-child(2) > .icondown').click()
        time.sleep(1)
        driver.find_element(By.LINK_TEXT, '带附件全回复').click()
        return '带附件全回复'
    except:
        driver.find_element(By.CSS_SELECTOR, '.toolbar > .u-btns:nth-child(2) > .u-btn:nth-child(2)').click()
        return '回复全部'



"""通用方法"""
from  six  import  wraps

def get_screen_add_report(driver, type_name):
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    file = path + "/report/%s" % (filename_time)
    name="/%s.png" % type_name
    if not os.path.isdir(file):
        os.makedirs(file)

    print(file)
    try:
        driver.get_screenshot_as_file(file+name)
        print("%s：截图成功！！！" % type_name)
    except BaseException as msg:
        print(msg)
    #driver.quit()



#截图装饰器调试
def get_screen_in_case_end_or_error(func):
    '''
    测试用例运行完成或者发生错误的时候进行截图
    :param func:
    :return:
    '''
    @wraps(func)
    def f1(driver,param):
        print(inspect.stack())
        try:
            func(driver,param)
            time.sleep(1)
            get_screen_add_report(driver,"运行成功")
        except:
            time.sleep(1)
            get_screen_add_report(driver,"运行失败截图")
            raise
    return f1


#@get_screen_in_case_end_or_error
def test_sendmail_page(driver,param):

    allure.dynamic.title("写信相关跳转，回复/转发仅支持阅读邮件页面")

    param["sendmail_page"]="转发"
    page_to=param["sendmail_page"]
    newwindow(driver,nb=0)
    if page_to=="写信":
        WebDriverWait(driver, 10,0.1).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR,".title"))).click()
    elif page_to=="发送":
        WebDriverWait(driver, 10,0.1).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(.,'发 送')]"))).click()
        #driver.find_element(By.XPATH, "//span[contains(.,'发 送')]").click()
    elif page_to=="回复":
        param["page_HF_type"]=page_HF(driver)
    elif page_to=="回复全部":
        param["page_HF_ALL_type"] = page_HF_ALL(driver)

    elif page_to == "转发":
        driver.find_element(By.CSS_SELECTOR,'.toolbar > .u-btn:nth-child(3)').click()
    else:
        pass

if __name__ == '__main__':
    pytest.main(["-s","test_sendmail_page.py::test_sendmail_page"])
