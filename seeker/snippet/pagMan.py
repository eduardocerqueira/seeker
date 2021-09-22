#date: 2021-09-22T16:59:48Z
#url: https://api.github.com/gists/261a3e56f0352cf36bdb0badcef18285
#owner: https://api.github.com/users/SU1199

from selenium import webdriver
import time
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import dropbox
from tinydb import TinyDB, Query

# Declare yo user
user = 'doom'
storeFolder = "sauce/"

driver = webdriver.Chrome('chromedriver.exe')
driver.get('https://www.anonigviewer.com/profile.php?u='+user)

# Upload to dropbox
uriList = []

def dropUpload(uris):
	dropbox_access_token= "" #Yo acess token goes here
	client = dropbox.Dropbox(dropbox_access_token)
	print("DataStore Connected")
	for uri in uris:
		to_path= "/"+uri
		from_path=uri
		client.files_upload(open(from_path, "rb").read(), to_path)
		print("Upload Komplete")


# Poor man's nosql store omegalul

def isPresent(url):
	db = TinyDB('db.json')
	p = Query()
	present = db.search(p.link==url)
	if(len(present)==0):
		db.insert({'link': url, 'timestamp': time.time()})
		return False
	else:
		return True

# Coutesy Of StackOverFlow lol

def download_file(url,type):
	if(isPresent(url)==False):
		timestr = time.strftime("%Y%m%d-%H%M%S")
		local_filename = storeFolder+timestr+type
		r = requests.get(url,verify=False, stream=True)
		with open(local_filename, 'wb') as f:
			for chunk in r.iter_content(chunk_size=1024): 
				if chunk:
					f.write(chunk)
		uriList.append(local_filename)
		return local_filename

# Ugly code don't look

wait = WebDriverWait(driver, 50)	# 50 Sec of summer
sauceCheck = wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[4]/div[3]/div/div[1]/div[1]/div/div')))
sauces = driver.find_elements_by_class_name('mb-4')
print(len(sauces))
i=1
preSauces = []
for sauce in sauces:
	preSauces.append(driver.find_element_by_xpath('/html/body/div[4]/div[3]/div/div[1]/div['+str(i)+']/div/div').get_attribute("onclick"))
	i=i+1

driver.close()

# Video or an Image ?

for preSauce in preSauces:
	if preSauce.find("GraphVideo")!=-1:
		type_ = ".mp4"
	elif preSauce.find("GraphImage")!=-1:
		type_ = ".jpg"
	link = re.findall(r'(https?://\S+)', preSauce)[0].split('"')[0]
	download_file(link,type_)

dropUpload(uriList)