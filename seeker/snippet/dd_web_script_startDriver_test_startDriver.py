#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
import allure
from selenium import webdriver
import requests
import json

def test_startdriver(param, driver):
    allure.dynamic.title("启动网页%s"%param['path'])
    path = param['path']
    # if True == param['isLogin']:
    #     sid = login_sid(param['username'], param['password'])
    #     path = path.replace('${sid}',sid)
    driver.get(path)



def login_sid(uid,password):
    url = 'http://cloud.mail.wo.cn/coremail/s/json?func=user:login'
    params = "{\"uid\":\"%s\",\"password\":\"%s\"}"%(uid,password)
    headers = {
    }
    res = requests.request('post', url, data=params, headers=headers)
    return res.json()['var']['sid']


# if __name__=="__main__":
#     login_sid('18707142515@wo.cn','Tsz201926')







