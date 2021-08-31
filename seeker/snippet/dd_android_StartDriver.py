#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# coding=utf-8
from appium import webdriver
import dill

class StartDriver:
    def __init__(self,platformName,platformVersion,deviceName,udid,port,path,noReset,appActivity):
        self.platformName = platformName
        self.platformVersion = platformVersion
        self.deviceName = deviceName
        self.udid = udid
        self.port = port
        self.path = path
        self.noReset = noReset
        self.appActivity = appActivity


    def startDriver(self):
        desired_caps = {}
        desired_caps['platformName'] = self.platformName
        desired_caps['platformVersion'] = self.platformVersion
        desired_caps['deviceName'] = self.deviceName
        desired_caps['udid'] = self.udid
        desired_caps['port'] = self.port
        desired_caps['path'] = self.path
        desired_caps['newCommandTimeout'] = "3600"
        desired_caps['noReset'] = self.noReset
        desired_caps['appPackage'] = "com.asiainfo.android"
        desired_caps['appActivity'] = self.appActivity
        # desired_caps['unicodeKeyboard'] = True
        # desired_caps['resetKeyboard'] = True
        # 启动app
        print("启动")
        driver = webdriver.Remote('http://localhost:%s/wd/hub'%self.port, desired_caps)
        set('driver', dill.dumps(driver))
        return driver

