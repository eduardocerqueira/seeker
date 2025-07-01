#date: 2025-07-01T17:14:47Z
#url: https://api.github.com/gists/eeb66b51aa549a27d712683eb7460a51
#owner: https://api.github.com/users/farzadoxo

import os
from colorama import Fore , init

init()

def lower_case(path:str) -> None:
    os.chdir(path=path)
    items = os.listdir(path=path)
    print(Fore.YELLOW+"OS path changed to : {}".format(os.getcwd() ))

    for i in items:
        if list(i)[0] != ".":
            if os.path.isdir("{}/{}".format(path,i)):
                print(Fore.BLUE+f"Directory Detected : {i}"+Fore.RESET)
                lower_case(path=f"{path}/{i}")
            
            if os.path.isfile("{}/{}".format(path,i)):
                os.rename(f"{path}/{i}",i.lower()) #only change lower to upper for uppering!
                print(Fore.LIGHTGREEN_EX+f"File renamed : {i} >>> {i.lower()}"+Fore.RESET)