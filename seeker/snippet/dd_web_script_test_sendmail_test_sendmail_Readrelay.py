#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#写信结果页面处理
from web.script.test_sendmail.test_sendmail_editor import *
def test_sendmail_Readrelay_a(driver, param):
    allure.dynamic.title("转发-写信页面检查，预期可见原邮件的主题与正文，收件人输入框为空")
    #当前用户
    name=param["name"]
    #原页面参数
    Read_fr=param["Read_fr"]
    Read_to_list=param["Read_to_list"]
    Read_subject=param["Read_subject"]
    Read_text=param["Read_text"]
    Read_FJ_list=param["Read_FJ_list"]
    #通过点击回复，新打开的写信页面参数：
    send_Read_to_str=param["send_Read_to_str"]
    send_Read_subject=param["send_Read_subject"]
    send_Read_text=param["send_Read_text"]
    send_Read_FJ_list=param["send_Read_FJ_list"]


    assert "Fw:" in send_Read_subject ,"转发的时候预期主题包含'Fw:'标识"
    assert send_Read_to_str==' ' ,"转发的时候收件人为空"
    assert Read_subject in send_Read_subject ,"转发的时候预期包含原主题"
    assert Read_text in send_Read_text,"转发页面预期包含原正文"
    assert "-----原始邮件-----" in send_Read_text,"转发页面文案包含原始邮件文案"
    assert Read_FJ_list==send_Read_FJ_list,"转发时，附件信息不一致"



if __name__ == '__main__':
    pytest.main(["-s","test_sendmail_Readrelay.py::test_sendmail_Readrelay_a"])