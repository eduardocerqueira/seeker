#date: 2021-08-31T03:11:53Z
#url: https://api.github.com/gists/62faf48c772279e74263efe5c9644541
#owner: https://api.github.com/users/gbrfilipe

import pyautogui

host = ""
user = ""
password = ""

code_to_run = "\copy covid19_al from '"+ full_output_path + "' WITH DELIMITER ',' CSV HEADER;"
images_path = cwd + "\\imagens\\"

pyautogui.PAUSE = 0.5
pyautogui.press('win')
pyautogui.typewrite('psql')
pyautogui.press('enter')
pyautogui.typewrite(host)
pyautogui.press('enter')
pyautogui.press('enter')
pyautogui.press('enter')
pyautogui.typewrite(user)
pyautogui.press('enter')
pyautogui.locateOnScreen(images_path + 'password_image.png')
pyautogui.typewrite(password)
pyautogui.press('enter')
pyautogui.locateOnScreen(images_path + 'warning_success_image.png')
pyautogui.typewrite(code_to_run)
pyautogui.press('enter')