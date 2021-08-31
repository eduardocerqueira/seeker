#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
from pymemcache.client.base import Client
import dill
import os
client = Client(('localhost', 11211))

if __name__=="__main__":
    client.set('param', dill.dumps( {'feature': '登录', 'story': '密码登录','path': 'https://mail.wo.cn/','driver':'','name':'18707142515','password':'Tsz201926'}))

    os.system('pytest -s script/startDriver/test_startDriver.py::test_startdriver --alluredir=report/tmp44')

    os.system('pytest -s script/login/test_login.py::test_name_password_login --alluredir=report/tmp44')
