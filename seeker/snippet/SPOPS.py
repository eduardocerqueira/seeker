#date: 2022-05-12T17:05:15Z
#url: https://api.github.com/gists/62b9cd82dd882fd085e707f99190a0c7
#owner: https://api.github.com/users/TheDucky

from bs4 import BeautifulSoup
import os
import string
import random

#MAX length of the randomly generated code
LEN = 6

ttlc = int(input("how many pictures do you want to download? "))

for x in range(ttlc):
    #Generate the code and join the web adress to the code
    ran = ''.join(random.choices(string.ascii_lowercase + string.digits, k = LEN))
    LINK = "https://prnt.sc/" + ran

    #Print the conformation
    print('[\033[0;32m========================================\033[0m]')
    print("Random code generated: [\033[0;32m{}\033[0m]".format(ran))
    print("Getting \033[0;32m{0}\033[0m src file.\n".format(LINK))

    #Download the src file to local system
    os.system("wget {}".format(LINK))

    #Open the file and pass it through bs4
    with open(ran, 'r') as rawHtml:
        htmlProcess = BeautifulSoup(rawHtml, "html.parser")

    #Look for the <img> tag
    imgTag = htmlProcess.img
    os.system("rm -r {}".format(ran))

    if "https://" not in imgTag['src']:
        print('Sorry, the attempted combination was not valid on \'prnt.sc\'')
        exit()
    else:
        print("Link to download image: \033[0;32m{}\033[0m\n".format(imgTag['src']))
        os.system("wget {}".format(imgTag['src']))