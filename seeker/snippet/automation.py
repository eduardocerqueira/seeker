#date: 2023-09-25T16:47:55Z
#url: https://api.github.com/gists/e59831c8af64b74d8efdc6c84292a127
#owner: https://api.github.com/users/LuisFelipeFrancisco

import pyautogui
import time

def cole_valores_na_aplicacao(valores):
    pyautogui.hotkey('alt', 'tab')
    time.sleep(1)

    for valor in valores:
        pyautogui.typewrite(str(valor))
        pyautogui.press('enter')
        time.sleep(0.100)
#=""""&A1&""""&","
valores = ["value1","value2"]
cole_valores_na_aplicacao(valores)