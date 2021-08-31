#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

from Util.AppiumExtend  import *
import dill
import os
from SQL import SQLUtil
from mainScript.android.Util.memcachedUtil import *
import allure



def startScript(id):

    useId = id
    id = 21
    print(useId)
    list = SQLUtil.SQLUtil('select teststep,appsteptype from apptest_appstep where Appcase_id = %s'%id).sqlExecute()
    for step in list:
        if 1 == step[1]:
            set('param', dill.dumps(step[0]))
        else :
            print('pytest -q %s --alluredir=../../static/report/tmp%s'%(step[0],id))
            os.system('pytest -q %s --alluredir=../../static/report/tmp%s'%(step[0],id))

    os.system('allure generate ../../static/report/tmp%s -o ../../static/report/report%s --clean'%(id,id))

    SQLUtil.SQLUtil('update apptest_appcase set apptestresult = 0 where id = %s'%useId).sqlExecute()
    # pytest.main(["-q","Script/android/Inbox/test_inbox.py::test_dropDownMenu","--alluredir=static/report/tmp2"])

    # set('param', dill.dumps({'feature': '收件箱', 'story': '过滤邮件:批量编辑:已读/未读', 'title': '', 'mailType': '批量编辑','id': 0}))
    # os.system('pytest -q Inbox/test_inbox.py::test_dropDownMenu --alluredir=report/tmp1')
    return True


if __name__ == "__main__":

    startScript(3)
