#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

import allure
import os

def test_adbShell(param):
    allure.dynamic.title("adb 工具")
    print(param['adb'])

    for shell in param['adb']:
        os.system(shell)

