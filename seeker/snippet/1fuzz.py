#date: 2023-10-31T16:40:35Z
#url: https://api.github.com/gists/9f8b3d009256fed9a8d6fcc592e06b88
#owner: https://api.github.com/users/33gl00

###############
# URL FUZZING #
###############

import requests
import os

import argparse
parser = argparse.ArgumentParser(
                    prog = '1fuzz',
                    description = 'fuzzing urls with list from file',
                    epilog = 'python3 1fuzz.py --url "https://eegloo.fr/?key=" --wordlist dico.txt --output fuzzReport.txt ')

parser.add_argument('-u', '--url', type=str, required=True, help="")
parser.add_argument('-w', '--wordlist', type=str, required=True, help="")
parser.add_argument('-o', '--output', type=str, required=False,  const="fuzzReport.txt")
input = parser.parse_args()
output = input.output
url = input.url
wordlist = input.wordlist

########################
# Reprint
lastPrintSize = 0
def reprint(txt, finish=False) :
	global lastPrintSize
	print(' '*lastPrintSize, end='\r')

	if finish:
		end = "\n"
		lastPrintSize = 0
	else : 
		end = "\r"
		lastPrintSize = len(txt)
	print(txt, end=end)
########################
def clear() :
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # macOS et Linux
        os.system("clear")
########################
def checkAndSave(url, count, total):
    response = requests.get(url)

    if response.text and (response.status_code != 404) :
        with open(output, "a") as fichier:
            fichier.write(f"{url} \n")
        return True
    else : 
        reprint(f"[-] {url} {count}/{total}")
        return False

clear()
print(url)
count = 0

with open(wordlist, "r") as file_a:
    allPattern = file_a.readlines()
    total_pattern = len(allPattern)
    for pattern in allPattern:
        count+=1
        link = f"{url}{pattern.strip()}"
        if checkAndSave(link, str(count), str(total_pattern)) :
            print(f"[+] {pattern.strip()} {str(count)}/{str(total_pattern)}")

print("D0ne!")
exit()

####### eegloo