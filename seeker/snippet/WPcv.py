#date: 2023-10-31T16:50:29Z
#url: https://api.github.com/gists/ec3da65664e7e4d06342b95587edfe4d
#owner: https://api.github.com/users/33gl00

#################################
# Detect CV file from wordPresS #
#################################

####################
#  ██████ ██    ██ #
# ██      ██    ██ #
# ██      ██    ██ #
# ██       █    █  #
#  ██████   ████   #
####################

import requests
import os
import time

import argparse
parser = argparse.ArgumentParser(
                    prog = 'download Url List',
                    description = 'download source from url file list',
                    epilog = 'python3 WPcv.py --cvlist CVlist.txt --wordpressurl https://www.society.fr/')

parser.add_argument('-u', '--wordpressurl', type=str, required=True, help="")
parser.add_argument('-c', '--cvlist', type=str, required=True, help="")
parser.add_argument('-o', '--output', type=str, required=False, const="CVreport.txt")
input = parser.parse_args()
wordpressUrl = input.wordpressurl
cvList = input.cvlist
report = input.output

extensions = ["pdf", "doc", "docx"]

years = ["2020","2021","2022","2023"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]

URL_buffer = []
outputFile=open(report,"w+")

# Word List
with open(cvList, "r") as patternList:
    fileNames = patternList.readlines()

# Supprimer les sauts de ligne et espaces en début/fin
fileNames = [name.strip() for name in fileNames]

def view(url, count) :
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # macOS et Linux
        os.system("clear")

    print(f"Hit : {url}")
    print(f"CV detected : {count}")

def checkCV(url):
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0", 'Accept': "*/*"}
    response = requests.get(url,  headers=headers)

    if response.status_code == 200:
        return True
    return False

def buildUrl(linkSeed,year) :
    special_chars = ["", " ", "-","_"]
    CVlinks = [linkSeed]

    for salt in special_chars :
        cvLink = f"{linkSeed}{salt}{year}"
        CVlinks.append(cvLink)
    return CVlinks

def downloadCV(file_path):
    # Ouvre le fichier contenant les urls
    with open(file_path, 'r') as file:
        urls = file.readlines()

    id = 1

    for url in urls:
        # Supprimer les sauts de ligne et espaces en début/fin
        url = url.strip()  

        file_name = url.split("/")[-1]
        file_name = f"{id}_{file_name}"
        id += 1

        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as output_file:
                output_file.write(response.content)
            print(f"Le fichier {file_name} a été téléchargé avec succès.")
        else:
            print(f"Échec du téléchargement du fichier depuis l'URL : {url}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
count = 0
for year in years:
    for month in months :
        for name in fileNames:
            cvLink = f"{wordpressUrl}/wp-content/uploads/{year}/{month}/{name}"
            LinksFormatted = buildUrl(cvLink, year)
            for link in LinksFormatted :
                for extension in extensions :
                    url = f"{link}.{extension}"
                    view(url, count)

                    if checkCV(url):
                        count = count + 1
                        outputFile.writelines(url + "\n")
                        
                        # Check doublons
                        for nb in range(3) :
                            if (nb != 0):
                                doublon = f"{link}-{nb}.{extension}"
                                if checkCV(doublon) :
                                    count = count + 1
                                    outputFile.writelines(doublon + "\n")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

downloadCV(outputFile)
view(wordpressUrl, count)
# eegloo