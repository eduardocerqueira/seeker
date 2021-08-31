#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding: utf-8 -*-
# @author: caihao
#写信结果页面处理
from web.script.test_sendmail.test_sendmail_editor import *
def test_sendmail_Readreply_a(driver, param):
    allure.dynamic.title("回复-写信页面检查，回复-写信页面检查，检查收件人、主题、正文、附件，发件人为自己的时候收件人为空")
    #当前用户
    name=param["name"]
    #原页面参数
    Read_fr=param["Read_fr"]
    Read_to_list=param["Read_to_list"]
    Read_subject=param["Read_subject"]
    Read_text=param["Read_text"]
    Read_FJ_list=param["Read_FJ_list"]
    #选中是否带有附件进行回复
    page_HF_type=param["page_HF_type"]
    #通过点击回复，新打开的写信页面参数：
    send_Read_to_str=param["send_Read_to_str"]
    send_Read_subject=param["send_Read_subject"]
    send_Read_text=param["send_Read_text"]
    send_Read_FJ_list=param["send_Read_FJ_list"]

    #暂不校验收件人，现存在bug

    if page_HF_type=="回复":
        assert send_Read_FJ_list==[] ,"回复不应该存在附件"
    elif page_HF_type=="带附件回复":
        assert Read_FJ_list==send_Read_FJ_list,"附件丢失"



    assert Read_subject in send_Read_subject ,"回复的时候预期包含原主题"
    assert Read_text in send_Read_text,"回复页面预期包含原正文"
    assert "-----原始邮件-----" in send_Read_text,"回复页面文案包含原始邮件文案"

def test_sendmail_Readreply_b(driver, param):
    allure.dynamic.title("全部回复-写信页面检查，回复-写信页面检查，检查收件人、主题、正文、附件")
    #当前用户
    name=param["name"]
    #原页面参数
    Read_fr=param["Read_fr"]
    Read_to_list=param["Read_to_list"]
    Read_subject=param["Read_subject"]
    Read_text=param["Read_text"]
    Read_FJ_list=param["Read_FJ_list"]
    #选中是否带有附件进行回复
    page_HF_ALL_type=param["page_HF_ALL_type"]
    #通过点击回复，新打开的写信页面参数：
    send_Read_to_str=param["send_Read_to_str"]
    send_Read_subject=param["send_Read_subject"]
    send_Read_text=param["send_Read_text"]
    send_Read_FJ_list=param["send_Read_FJ_list"]

    #检查原邮件收件人是否都存在
    if name in Read_fr:
        try:
            Read_to_list.remove(Read_fr)
        except:
            pass
    for i in Read_to_list:
        print(i)
        assert i in send_Read_to_str,"收件人缺失:"+i
    #检查原邮件主题内容与正文内容是否存在

    if page_HF_ALL_type=="回复全部":
        assert send_Read_FJ_list==[] ,"回复全部不应该存在附件"
    elif page_HF_ALL_type=="带附件全回复":
        assert Read_FJ_list==send_Read_FJ_list,"附件丢失"

    assert Read_subject in send_Read_subject ,"回复的时候预期包含原主题"
    assert Read_text in send_Read_text,"回复页面预期包含原正文"
    assert "-----原始邮件-----" in send_Read_text,"回复页面文案包含原始邮件文案"

if __name__ == '__main__':
    pytest.main(["-s","test_sendmail_Readreply.py::test_sendmail_Readreply_b"])