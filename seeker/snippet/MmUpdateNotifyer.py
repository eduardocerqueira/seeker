#date: 2022-06-22T17:20:42Z
#url: https://api.github.com/gists/c8b1e5fd6dad5d4ca50b2366b3cb2ad6
#owner: https://api.github.com/users/elpatron68

#!/bin/python3
# requires packaging and requests-html
# run scheduled (@daily) with cron
import re
from urllib import response
import requests
from os.path import exists
from requests_html import HTMLSession
from packaging import version

WEBHOOKURL = 'https://mattermost.yourdomain.com/hooks/your_webhook'
CHANNEL = 'chennel name'

def getLatestVersion():
    downloadUrl = ""

    session = HTMLSession()
    r = session.get('https://mattermost.com/deploy/')
    htmlPageText=r.text
    
    regex = r'https:\/\/releases\.mattermost\.com\/\d+\.\d+\.\d+\/mattermost-\d+\.\d+\.\d+-linux-amd64\.tar\.gz'
    downloadUrl = ''
    version = ''

    try:
        downloadUrl = re.findall(regex, htmlPageText)[0]
    except:
        pass
    
    try:
        version = re.findall(r'\d+\.\d+\.\d+', downloadUrl)[0]
    except:
        pass
        
    return downloadUrl, version
    
def readLastversion():
    with open('lastversion.txt', 'r') as file:
        result = file.read().rstrip()
    return result

def writeLastversion(version):
    with open('lastversion.txt', 'w') as file:
        file.write(version)
    
def isNewer(latestVersion, lastVerion):
    return version.parse(latestVersion) > version.parse(lastVerion)

def sendMM(text):
    headers = {'Content-Type': 'application/json',}
    values = '{ "channel": "' + CHANNEL + '", "text": "' + text + '"}'
    response = requests.post(WEBHOOKURL, headers=headers, data=values)
    return response.status_code
    
if __name__ == "__main__":
    if not exists('lastversion.txt'):
        writeLastversion("0.0.0")
        
    url, ver = getLatestVersion()
    lv = readLastversion()
    
    if (isNewer(ver, lv)):
        writeLastversion(ver)
        print('New Mattermost version found, information updated:')
        print('Former version: ' + lv)
        print('Latest version: ' + ver)
        print('Download URL:   ' + url)
        text = 'New Mattermost version found!\nLatest version: ' + ver + '.\nFormer version: ' + lv + '\nDownload URL: ' + url + '\n[Release notes](https://docs.mattermost.com/install/self-managed-changelog.html)'
        result = sendMM(text=text)
        print('Message sent: ' + str(result))
        
    else:
        print('Nothing to do (you are up-to-date).')