#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

from memcachedUtil import *
import pickle
import os

class ExecutePytest:

    def __init__(self,parm,case):
        self.param = parm
        self.case = case

    def executeCase(self):
        set('param', pickle.dumps(self.param))
        os.system('pytest -s %s --alluredir=report/tmp'%self.case)

