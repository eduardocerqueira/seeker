#date: 2023-01-03T16:29:15Z
#url: https://api.github.com/gists/665b57499bb3f2c33d40b686fe278a50
#owner: https://api.github.com/users/Sematemur

from selenium.webdriver.common.by import By
from bilgi import isim, sifre,hashtag
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
class Twitter:
    def __init__(self,isim,sifre):
                yol = "C:\\Users\\SEMA\\Downloads\\chromedriver_win32\\chromedriver.exe"
                self.driver = webdriver.Chrome(yol)
                self.isim = isim
                self.sifre = sifre
                time.sleep(3)
    def girisyapma(self):
        self.driver.get("https://twitter.com/i/flow/login")
        time.sleep(4)
        isimyolu=self.driver.find_element(By.XPATH,"//*[@id='layers']/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input")
        isimyolu.send_keys(self.isim)
        isimyolu.send_keys(Keys.ENTER)
        time.sleep(2)
        sifreyolu=self.driver.find_element(By.XPATH,"//*[@id='layers']/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input")
        sifreyolu.send_keys(self.sifre)
        time.sleep(3)
        sifreyolu.send_keys(Keys.ENTER)
        time.sleep(3)
    def bildirimlerigösterme(self):
        self.driver.find_element(By.XPATH,"//*[@id='react-root']/div/div/div[2]/header/div/div/div/div[1]/div[2]/nav/a[3]").click()
        time.sleep(3)
    def mesajlarigösterme (self):
        self.driver.find_element(By.XPATH,"//*[@id='react-root']/div/div/div[2]/header/div/div/div/div[1]/div[2]/nav/a[4]").click()
        time.sleep(4)
    def hastaghgörearama(self,hashtag):
        self.hashtag=hashtag
        arama=self.driver.find_element(By.XPATH,"//*[@id='react-root']/div/div/div[2]/main/div/div/div/div[2]/div/div[2]/div/div/div/div[1]/div/div/div/form/div[1]/div/div/div/label/div[2]/div/input")
        arama.send_keys(self.hashtag)
        arama.send_keys(Keys.ENTER)
        time.sleep(4)
    def takipetme(self):
        self.driver.find_element(By.XPATH,"//*[@id='react-root']/div/div/div[2]/main/div/div/div/div[1]/div/div[3]/div/section/div/div/div[3]/div/div/div/div/div[2]/div[1]/div[2]/div").click()
        time.sleep(3)




twitter=Twitter(isim,sifre)
twitter.girisyapma()
twitter.mesajlarigösterme()
twitter.bildirimlerigösterme()
twitter.hastaghgörearama(hashtag)
twitter.takipetme()



