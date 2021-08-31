#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

from memcachedUtil import *
import dill
import os
from SQL import SQLUtil
import pytest
from AppiumExtend import Appium_Extend

def startScript(id):

    list = SQLUtil.SQLUtil('select teststep,appsteptype from apptest_appstep where Appcase_id = %s'%id).sqlExecute()
    for step in list:
        if 1 == step[1]:
            pass
            set('param', dill.dumps(step[0]))
        else :
            print(step[0])
            pytest.main(["-q",step[0],"--alluredir=static/report/tmp21"])
            # print('pytest -q mainScript/android/%s --alluredir=static/report/tmp%s'%(step[0],id))
            # os.system('pytest -q mainScript/android/%s --alluredir=static/report/tmp%s'%(step[0],id))

    # pytest.main(["-q","Script/android/Inbox/test_inbox.py::test_dropDownMenu","--alluredir=static/report/tmp2"])

    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '批量编辑','id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp1')
    return True


if __name__ == '__main__':
    # startScript(21)

    # 截图保留
    param = dill.loads(get('param'))
    driver = dill.loads(get('driver%s' % param['udid']))
    extend = Appium_Extend(driver)
    abstract = driver.find_element_by_id('com.asiainfo.android:id/iv_unread')
    extend.get_screenshot_by_element(abstract).write_to_file('/Users/kosenmac1/Downloads/image/', 'reply')

     # while True:
    # startDriverLogin = StartDriver('Android', '5.1.1', 'Android', '753f68a10404', '4725',
    #                      'C:\\Users\\tsz\\Downloads\\沃邮箱_8.2.4.apk',
    #                      False,'com.asiainfo.mail.ui.mainpage.SplashActivity')
    #
    # startDriverMailList = StartDriver('Android', '5.1.1', 'A  ndroid', '753f68a10404', '4725',
    #                      'C:\\Users\\tsz\\Downloads\\沃邮箱_8.2.4.apk',
    #                      True,'')
    #
    # driver = startDriverLogin.startDriver()

     # driver.keyevent()

    #####################################
    # set('param', dill.dumps(
    #     {'feature': '收件箱埋数', 'story': '密码登录', 'title': '', 'mailType': '', 'phno': '18707142515',
    #      'pwd': 'tsz@201926'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp44')
    # os.system('pytest -q Login/test_login.py::test_choiceQQ --alluredir=report/tmp44')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp44')
    # os.system('allure generate report/tmp44 -o report/report44 --clean')

    #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱登录163', 'story': '密码登录', 'title': '', 'mailType': '', 'phno': 'womail_test',
    #      'pwd': '1qa2ws'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp43')
    # os.system('pytest -q Login/test_login.py::test_choice163 --alluredir=report/tmp43')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp43')
    # os.system('allure generate report/tmp43 -o report/report43 --clean')

    # #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱快速重置密码server报错', 'story': '重置密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': 'QShJFFS8j'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp42')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp42')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp42')
    # os.system('pytest -q Login/test_login.py::test_clickConfirm --alluredir=report/tmp42')
    # os.system('pytest -q Login/test_login.py::test_clickKeyLoginOffData --alluredir=report/tmp42')
    # os.system('pytest -q PhoneCase/test_phoneCase.py::test_onData --alluredir=report/tmp42')
    # os.system('pytest -q Login/test_login.py::test_checkUnLoginServer --alluredir=report/tmp42')
    # os.system('pytest -q Setting/test_setting.py::test_login --alluredir=report/tmp42')
    # os.system('allure generate report/tmp42 -o report/report42 --clean')

    # #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱快速重置密码', 'story': '重置密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '123456'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp41')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp41')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp41')
    # os.system('pytest -q Login/test_login.py::test_clickConfirm --alluredir=report/tmp41')
    # os.system('pytest -q Login/test_login.py::test_clickKeyLogin --alluredir=report/tmp41')
    # os.system('allure generate report/tmp41 -o report/report41 --clean')


    #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱验证码登录server错误', 'story': '验证码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '123456'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp40')
        # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp40')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp40')
    # os.system('pytest -q Login/test_login.py::test_resetPwdLoginAndSwitchWLAN --alluredir=report/tmp40')
    # os.system('allure generate report/tmp40 -o report/report40 --clean')

    # #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱验证码登录错误验证码', 'story': '验证码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '123456'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp39')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp39')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp39')
    # os.system('pytest -q Login/test_login.py::test_loginCode --alluredir=report/tmp39')
    # os.system('pytest -q Login/test_login.py::test_resetPwdLogin --alluredir=report/tmp39')
    # os.system('allure generate report/tmp39 -o report/report39 --clean')

    #####################################
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱验证码登录错误手机号', 'story': '验证码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': ''}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp38')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp38')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp38')
    # os.system('pytest -q Login/test_login.py::test_getCode --alluredir=report/tmp38')
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱验证码登录错误手机号', 'story': '验证码登录', 'title': '', 'mailType': '', 'phno': '18600113697',
    #      'pwd': ''}))
    # os.system('pytest -q Login/test_login.py::test_fillPhone --alluredir=report/tmp38')
    # os.system('pytest -q Login/test_login.py::test_clickLogin --alluredir=report/tmp38')
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱验证码登录错误手机号', 'story': '验证码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': ''}))
    # os.system('pytest -q Login/test_login.py::test_fillPhone --alluredir=report/tmp38')
    # os.system('pytest -q Login/test_login.py::test_clickLogin --alluredir=report/tmp38')
    # os.system('allure generate report/tmp38 -o report/report38 --clean')

    #####################################
    # client.set('param', dill.dumps({'feature': '登录', 'story': '验证码登录', 'title': '', 'mailType': '','phno':'18600113647','pwd':'1qa2ws'}))
    # set('param', dill.dumps({'feature': '精品沃邮箱验证码登录', 'story': '验证码登录', 'title': '', 'mailType': '','phno':'18600113647','pwd':'1qa2ws'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp37')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp37')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp37')
    # os.system('pytest -q Login/test_login.py::test_resetPwdLogin --alluredir=report/tmp37')
    # os.system('allure generate report/tmp37 -o report/report37 --clean')

    #####################################
    # startDriverLogin.startDriver()
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱登录失败服务器提示', 'story': '账号密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '123'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp36')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp36')
    # os.system('pytest -q Login/test_login.py::test_loginAndSwitchData --alluredir=report/tmp36')
        # os.system('pytest -q Login/test_login.py::test_checkUnLoginServer --alluredir=report/tmp36')
    # set('param', dill.dumps(
    #         {'feature': '精品沃邮箱登录失败服务器提示', 'story': '账号密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #          'pwd': '05Gx6YDq'}))
    # os.system('pytest -q Setting/test_setting.py::test_login --alluredir=report/tmp36')
    # os.system('allure generate report/tmp36 -o report/report36 --clean')

    #####################################
    # startDriverLogin.startDriver()
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱登录失败app提示', 'story': '账号密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '123'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp35')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp35')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp35')
    # os.system('pytest -q Login/test_login.py::test_checkUnLoginApp --alluredir=report/tmp35')
    # set('param', dill.dumps(
    #     {'feature': '精品沃邮箱登录失败app提示', 'story': '账号密码登录', 'title': '', 'mailType': '', 'phno': '18600113647',
    #      'pwd': '05Gx6YDq'}))
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp35')
    # os.system('allure generate report/tmp35 -o report/report35 --clean')
    #####################################
    # startDriverLogin.startDriver()
    # set('param', dill.dumps({'feature': '精品沃邮箱登录', 'story': '账号密码登录', 'title': '', 'mailType': '','phno':'18600113647','pwd':'111qqqaaA'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp34')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp34')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp34')
    # os.system('allure generate report/tmp34 -o report/report34 --clean')

    #####################################
    # set('param', dill.dumps({'feature': '登录', 'story': '验证码登录', 'title': '', 'mailType': '','phno':'womail_test@163.com','pwd':'1qa2ws'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp30')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp30')
    # os.system('pytest -q Login/test_login.py::test_resetPwd --alluredir=report/tmp30')
    # os.system('pytest -q Login/test_login.py::test_resetPwdLogin --alluredir=report/tmp30')
    # os.system('allure generate report/tmp30 -o report/report30 --clean')
    # startDriverLogin.startDriver()

    # ####################################
    # set('param', dill.dumps({'feature': '登录收件箱', 'story': '登录收件箱邮件检查', 'title': '', 'mailType': '','id':0,'phno':'18707142515','pwd':'Tsz@201926'}))
    # os.system('pytest -q Login/test_login.py::test_agreePrivacy --alluredir=report/tmp32')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp32')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp32')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMail --alluredir=report/tmp32')
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp32')
    # # os.system('pytest -q MailDetail/test_mailDetail.py::test_checkMailDetail --alluredir=report/tmp32')
    # # startDriverLogin.startDriver()
    #
    # ######################################
    # # client.set('param', dill.dumps({'feature': '设置', 'story': '添加账号', 'title': '', 'mailType': '','text':'添加邮箱','phno':'18010096059','pwd':'CUIHN0506@'}))
    # set('param', dill.dumps({'feature': '设置', 'story': '添加账号', 'title': '', 'mailType': '','text':'添加邮箱','phno':'18010096059','pwd':'CUIHN0506@'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp31')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickSetting --alluredir=report/tmp31')
    # os.system('pytest -q Setting/test_setting.py::test_clickText --alluredir=report/tmp31')
    # os.system('pytest -q Login/test_login.py::test_choiceWo --alluredir=report/tmp31')
    # os.system('pytest -q Login/test_login.py::test_login --alluredir=report/tmp31')
    # # os.system('allure generate report/tmp31 -o report/report31 --clean')


    # pytest.main(["-s", 'Inbox/test_inbox.py::test_countList', '--alluredir', 'report/tmp'])

    # set('param', dill.dumps({'feature': '去root警告', 'story': '去root警告', 'title': '','mailType':''}))
    # os.system('pytest -s Inbox/test_inbox.py::test_clickOk --alluredir=report/tmp')

    # driver = dill.loads(get('driver'))
    # extend = Appium_Extend(driver)
    # star = driver.find_element_by_id('com.asiainfo.android:id/iv_star')
    # extend.get_screenshot_by_element(star)
    # extend.write_to_file("d:\\screen\\","detail_star")

    #########################################
    #
    # textList = [
    #         'batch_unread_star_sended_1',
    #         'batch_unread_star_sended_2',
    #         'batch_deleted_2',
    #         'batch_deleted_1',
    #         # 'batch_more_deleted_2',
    #         # 'batch_more_deleted_1',
    #         # 'batch_one_rejection',
    #         # 'left_delete',
    #         # 'right_delete',
    #         # 'right_rejection',
    #         # 'right_move_sended',
    #         # 'right_move_deleted',
    #         # 'batch_one_agency_star',
    #         # 'reply_forward',
    #         # 'forward',
    #         # 'right_reply',
    #         # 'search',
    #         # 'count6',
    #         # 'count7',
    #         # 'count8',
    #         # 'count9',
    #         # 'count10',
    #         # 'count11',
    #         # 'count12',
    #         # 'count13',
    #         # 'count14',
    #         # 'count15',
    #         # 'count16',
    #         # 'count17',
    #         # 'count18',
    #         # 'count19',
    #         # 'count20',
    #         # 'count21',
    #         ]
    #
    # relist = textList[::-1]
    #
    # for i in relist:
    #     set('param', dill.dumps({'feature': '发件', 'story': '发邮件', 'title': '', 'mailType': '','id': 0,'text':i}))
    #     os.system('pytest -q Inbox/test_inbox.py::test_clickWrite --alluredir=report/tmp')
    #     os.system('pytest -q Outbox/test_outBox.py::test_writeMail --alluredir=report/tmp')
    #     os.system('pytest -q Outbox/test_outBox.py::test_send --alluredir=report/tmp')

    ######################################
    # client.set('param', dill.dumps({'feature': '设置', 'story': '添加邮件签名', 'title': '', 'mailType': '','text':'邮件签名','name':'测试姓名',
    #                          'mailbox':'test@wo.cn','phno':'13000000000','cpy':'测试公司','titl':'测试职衔','adr':'测试地址','fax':'测试传真'}))
    # set('param', dill.dumps({'feature': '设置', 'story': '添加邮件签名', 'title': '', 'mailType': '','text':'邮件签名','name':'测试姓名',
    #                          'mailbox':'test@wo.cn','phno':'13000000000','cpy':'测试公司','titl':'测试职衔','adr':'测试地址','fax':'测试传真'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp32')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickSetting --alluredir=report/tmp32')
    # os.system('pytest -q Setting/test_setting.py::test_clickText --alluredir=report/tmp32')
    # os.system('pytest -q Setting/test_setting.py::test_addBusinessCard --alluredir=report/tmp32')
    # os.system('pytest -q Setting/test_setting.py::test_back --alluredir=report/tmp32')
    # os.system('allure generate report/tmp32 -o report/report32 --clean')

    # ######################################
    # set('param', dill.dumps({'feature': '设置', 'story': '切换账号', 'title': '', 'mailType': '','text':'18707142515'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp33')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp33')
    # set('param', dill.dumps(
    #      {'feature': '设置', 'story': '切换账号', 'title': '', 'mailType': '', 'text': '收件箱'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickBacking --alluredir=report/tmp33')

    #####################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '批量编辑','id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp1')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp1')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp1')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '已读', 'id': 0, 'text': '标记已读'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp1')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp1')
    # set('param',
    #     dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '已读', 'id': 1, 'text': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp1')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp1')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp1')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp1')
    # set('param',
    #     dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '未读', 'id': 0, 'text': '标记未读'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp1')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp1')
    # set('param',
    #     dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '未读', 'id': 1, 'text': '标记未读'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp1')
    # # #######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp2')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp2')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp2')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 0, 'text': '标记星标'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp2')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp2')
    # set('param',
    #     dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '未读', 'id': 1, 'text': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp2')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp2')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp2')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp2')
    # set('param',
    #     dill.dumps(
    #         {'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 0, 'text': '取消星标'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp2')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp2')
    # set('param',
    #     dill.dumps(
    #         {'feature': '收件箱', 'story': '过滤邮件:批量编辑:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 1, 'text': '取消星标'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp2')
    # #################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp3')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp3')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp3')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '移动到'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp3')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '已发送'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp3')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '移动'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp3')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp3')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '已发送'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp3')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:过滤邮件:批量编辑:移动到已发送', 'title': '', 'mailType': '', 'id': 0,  'text': '已发送', 'findMail': 'batch_unread_star_sended_1'}))
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp3')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp3')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp3')
    # ######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp4')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp4')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp4')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '移动到'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp4')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp4')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '移动'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp4')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp4')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp4')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除',
    #      'findMail': 'deleted_2'}))
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp4')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp4')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp4')
    # #######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp5')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp5')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp5')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 0, 'text': '更多'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp5')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 0, 'text': '删除'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp5')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 0, 'text': '删除'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp5')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp5')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp5')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:更多删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除',
    #      'findMail': 'batch_deleted_1'}))
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp5')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp5')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp5')
    # #######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条拒收', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp6')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp6')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '更多'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp6')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '拒收'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp6')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '确定'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp6')
    # ########################################
    # set('param',dill.dumps({'feature': '收件箱', 'story': '左滑删除', 'title': '', 'mailType': '', 'id': 0,'text':'已删除','findMail':'left_delete'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_leftSlipDelete --alluredir=report/tmp7')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp7')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp7')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp7')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp7')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp7')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '右滑闪操删除', 'title': '', 'mailType': '删除', 'id': 0, 'text': '已删除'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp8')
    # set('param', dill.dumps(
    #         {'feature': '收件箱', 'story': '右滑闪操删除', 'title': '', 'mailType': '', 'id': 0, 'text': '删除'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp8')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操删除', 'title': '', 'mailType': '', 'id': 0, 'text': '确定'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp8')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp8')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除',
    #      'findMail': 'right_delete'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp8')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp8')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp8')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp8')
    # #######################################
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '拒收', 'findMail': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp9')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp9')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '确定', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp9')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp9')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操拒收', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除', 'findMail': 'right_rejection'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp9')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp9')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp9')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp9')
    # #######################################
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '移动到', 'findMail': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp10')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp10')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '已发送', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp10')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '移动', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp10')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp10')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已发送', 'title': '', 'mailType': '', 'id': 0, 'text': '已发送', 'findMail': 'right_move_sended'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp10')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp10')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp10')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp10')
    # #######################################
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '移动到', 'findMail': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp11')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp11')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp11')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '移动', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp11')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp11')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操移动到已删除', 'title': '', 'mailType': '', 'id': 0, 'text': '已删除', 'findMail': 'right_move_deleted'}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp11')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find --alluredir=report/tmp11')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp11')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp11')
    # #######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条转为待办', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp12')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp12')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条转为待办', 'title': '', 'mailType': '', 'id': 0, 'text': '更多'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp12')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '过滤邮件:批量编辑:单条转为待办', 'title': '', 'mailType': '', 'id': 0, 'text': '待办'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickContainsTest --alluredir=report/tmp12')
    # os.system('pytest -q AgencyList/test_agencyList.py::test_complete --alluredir=report/tmp12')
    # os.system('pytest -q AgencyList/test_agencyList.py::test_back --alluredir=report/tmp12')
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp12')
    #######################################
    # set('param', dill.dumps(
    #         {'feature': '收件箱', 'story': '右滑闪操:标为已读/未读', 'title': '', 'mailType': '已读', 'id': 0, 'text': '设为已读','findMail':''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp13')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp13')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp13')
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp13')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:标为已读/未读', 'title': '', 'mailType': '未读', 'id': 0, 'text': '设为未读','findMail':''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp13')
    # ########################################
    # set('param', dill.dumps(
    #         {'feature': '收件箱', 'story': '右滑闪操:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 0, 'text': '标记星标','findMail':''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp14')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp14')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp14')
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp14')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 0, 'text': '取消星标','findMail':''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp14')
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp14')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:标记/取消星标', 'title': '', 'mailType': '星标', 'id': 0, 'text': '标记星标','findMail':''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp14')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '回复', 'title': '', 'mailType': '回复', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp15')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_replyMailDetail --alluredir=report/tmp15')
    # os.system('pytest -q Outbox/test_outBox.py::test_send --alluredir=report/tmp15')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp15')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp15')
    # #########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '回复转发', 'title': '', 'mailType': '回复转发','id':1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp16')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_transferMailDetail --alluredir=report/tmp16')
    # os.system('pytest -q Outbox/test_outBox.py::test_inputSender --alluredir=report/tmp16')
    # os.system('pytest -q Outbox/test_outBox.py::test_send --alluredir=report/tmp16')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp16')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp16')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '转发', 'title': '', 'mailType': '转发','xpath':'','value':'','id':2}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp17')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_transferMailDetail --alluredir=report/tmp17')
    # os.system('pytest -q Outbox/test_outBox.py::test_inputSender --alluredir=report/tmp17')
    # os.system('pytest -q Outbox/test_outBox.py::test_send --alluredir=report/tmp17')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp17')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp17')
    ######################################
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:闪电回复', 'title': '', 'mailType': '回复', 'id': 3, 'text': '闪电回复', 'findMail': ''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_rightSlip --alluredir=report/tmp18')
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp18')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:闪电回复', 'title': '', 'mailType': '回复', 'id': 3, 'text': '好的,谢谢!', 'findMail': ''}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp18')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMailIcon --alluredir=report/tmp18')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp18')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:闪电回复', 'title': '', 'mailType': '回复', 'id': 3, 'text': '已发送', 'findMail': ''}))
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickText --alluredir=report/tmp18')
    # os.system('pytest -q DeleteList/test_deleteList.py::test_find_click --alluredir=report/tmp18')
    # set('param', dill.dumps(
    #     {'feature': '收件箱', 'story': '右滑闪操:闪电回复', 'title': '', 'mailType': '回复', 'id': 3, 'text': '好的,谢谢!',
    #      'findMail': 'right_reply'}))
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_find --alluredir=report/tmp18')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp18')
    # os.system('pytest -q Inbox/test_inbox.py::test_openSidebar --alluredir=report/tmp18')
    # os.system('pytest -q Sidebar/test_sidebar.py::test_clickInbox --alluredir=report/tmp18')
    # ######################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '','mailType':'','text':'search'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickSearch --alluredir=report/tmp19')
    # os.system('pytest -q SearchList/test_searchList.py::test_inputSearch --alluredir=report/tmp19')
    # os.system('pytest -q SearchList/test_searchList.py::test_search --alluredir=report/tmp19')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '', 'mailType': '', 'text': '所有'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp19')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '', 'mailType': '', 'text': '发件人'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp19')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '', 'mailType': '', 'text': '收件人'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp19')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '', 'mailType': '', 'text': '主题'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp19')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '搜索', 'title': '', 'mailType': '', 'text': '取消'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp19')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '列表加载数量', 'title': '','mailType':''}))
    # os.system('pytest -q Inbox/test_inbox.py::test_countList --alluredir=report/tmp20')
    ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:未读', 'title': '','mailType':'未读'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp21')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp21')
    # os.system('pytest -q Inbox/test_inbox.py::test_listTypeCheck --alluredir=report/tmp21')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:星标', 'title': '', 'mailType': '星标'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp22')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp22')
    # os.system('pytest -q Inbox/test_inbox.py::test_listTypeCheckStart --alluredir=report/tmp22')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:全部', 'title': '', 'mailType': '全部'}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp23')
    # os.system('pytest -q Inbox/test_inbox.py::test_screenMail --alluredir=report/tmp23')
    # ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '长按点击', 'title': '', 'mailType': '批量编辑', 'id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_longPressMail --alluredir=report/tmp24')
    # set('param', dill.dumps({'feature': '收件箱', 'story': '长按点击', 'title': '', 'mailType': '', 'id': 0, 'text': '取消'}))
    # os.system('pytest -q FlashMenu/test_flashMenu.py::test_clickTest --alluredir=report/tmp24')
    ########################################
    # set('param', dill.dumps({'feature': '收件箱', 'story': '邮件查看前后比对', 'title': '', 'mailType': '','id':1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMail --alluredir=report/tmp25')
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp25')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_checkMailDetail --alluredir=report/tmp25')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp25')
    # os.system('pytest -q Inbox/test_inbox.py::test_checkMail --alluredir=report/tmp25')
    ########################################
    # set('param', dill.dumps({'feature': '邮件详情', 'story': '详情星标/取消', 'title': '', 'mailType': '星标', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp26')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_star --alluredir=report/tmp26')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_check_star --alluredir=report/tmp26')
    # set('param', dill.dumps({'feature': '邮件详情', 'story': '详情星标/取消', 'title': '', 'mailType': '非星标', 'id': 1}))
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_star --alluredir=report/tmp26')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_check_star --alluredir=report/tmp26')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp26')
    # os.system('allure generate report/tmp26 -o report/report26 --clean')
    ########################################
    # set('param', dill.dumps({'feature': '邮件详情', 'story': '详情改变文本大小', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp27')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_textChange --alluredir=report/tmp27')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_slideTextChange --alluredir=report/tmp27')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp27')
    # os.system('allure generate report/tmp27 -o report/report27 --clean')
    ########################################
    # set('param', dill.dumps({'feature': '邮件详情', 'story': '详情全部回复', 'title': '', 'mailType': '', 'id': 1}))
    # os.system('pytest -q Inbox/test_inbox.py::test_clickMail --alluredir=report/tmp28')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_replyAll --alluredir=report/tmp28')
    # os.system('pytest -q MailDetail/test_mailDetail.py::test_backMailDetail --alluredir=report/tmp28')
    # os.system('allure generate report/tmp28 -o report/report28 --clean')


    # driver.quit()
    # os.system('allure generate report/tmp -o D:/report/report --clean')


    # os.system('allure serve report/tmp31')

    print('执行结束')

    # textList = [
    #         'batch_unread_star_sended_1',
    #         'batch_unread_star_sended_2',
    #         'batch_deleted_2',
    #         'batch_deleted_1',
    #         # 'batch_more_deleted_2',
    #         # 'batch_more_deleted_1',
    #         # 'batch_one_rejection',
    #         # 'left_delete',
    #         # 'right_delete',
    #         # 'right_rejection',
    #         # 'right_move_sended',
    #         # 'right_move_deleted',
    #         # 'batch_one_agency_star',
    #         # 'reply_forward',
    #         # 'forward',
    #         # 'right_reply',
    #         # 'search',
    #         # 'count6',
    #         # 'count7',
    #         # 'count8',
    #         # 'count9',
    #         # 'count10',
    #         # 'count11',
    #         # 'count12',
    #         # 'count13',
    #         # 'count14',
    #         # 'count15',
    #         # 'count16',
    #         # 'count17',
    #         # 'count18',
    #         # 'count19',
    #         # 'count20',
    #         # 'count21',
    #         ]
    #
    # relist = textList[::-1]
    #
    # for i in relist:
    #     set('param', dill.dumps({'feature': '发件', 'story': '发邮件', 'title': '', 'mailType': '','id': 0,'text':i}))
    #     os.system('pytest -q Inbox/test_inbox.py::test_clickWrite --alluredir=report/tmp')
    #     os.system('pytest -q Outbox/test_outBox.py::test_writeMail --alluredir=report/tmp')
    #     os.system('pytest -q Outbox/test_outBox.py::test_send --alluredir=report/tmp')
