#date: 2021-09-01T16:58:39Z
#url: https://api.github.com/gists/2065486850db85f1c00248b84239d0c8
#owner: https://api.github.com/users/heywoodlh

import subprocess
import sys
import pathlib
import zipfile
import urllib.request
import shutil

## Set this to the desired version for Winlogbeat
winlogbeat_version = '7.14.0'

## Set this to wherever your winlogbeat.yml is stored
winlogbeat_yml_url = 'http://localhost/winlogbeat.yml'

temp_dir = str(pathlib.Path.home()) + '\AppData\Local\Temp'

## Download Winlogbeat zip
print('Downloading Winlogbeat version ' + winlogbeat_version + ' zip file')

url = 'https://artifacts.elastic.co/downloads/beats/winlogbeat/winlogbeat-' + winlogbeat_version + '-windows-x86_64.zip'
outfile = temp_dir + '\winlogbeat.zip'

urllib.request.urlretrieve(url, outfile)

## Extract Winlogbeat zip
print("Extracting winlogbeat zip archive")
try: 
    with zipfile.ZipFile(outfile) as z:
        z.extractall(temp_dir)
except:
    print("Error encountered")
    sys.exit(1)

## Move output directory to C:\Program Files\Winlogbeat
outdir = temp_dir + "/winlogbeat-" + winlogbeat_version + "-windows-x86_64"
destdir = "C:\Program Files\Winlogbeat"
shutil.move(outdir,destdir)

## Install the winlogbeat service
print("Installing the winlogbeat service")
subprocess.call(["Powershell.exe", "-ExecutionPolicy", "Bypass", "-File", destdir + "\install-service-winlogbeat.ps1"])

## Stop winlogbeat
print("Stopping winlogbeat (in case it is already started)")
subprocess.call(["Powershell.exe", "Stop-Service", "winlogbeat"])

## Download the new winlogbeat.yml
print("Updating winlogbeat configuration file")
url = winlogbeat_yml_url
outfile = destdir + "\winlogbeat.yml"
urllib.request.urlretrieve(url, outfile) 

## Set winlogbeat to start automatically
print("Set winlogbeat to start on boot")
subprocess.call(["Powershell.exe", "Set-Service", "-Name", "winlogbeat", "-StartupType", "Automatic"])

## Start winlogbeat
print("Starting winlogbeat")
subprocess.call(["Powershell.exe", "Restart-Service", "winlogbeat"])

