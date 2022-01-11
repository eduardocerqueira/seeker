#date: 2022-01-11T17:16:59Z
#url: https://api.github.com/gists/9b05c3df72f5a6e2d90ec39cce729e2f
#owner: https://api.github.com/users/ziplex

## YOUTUBE VIDEO PART I: https://youtu.be/iZzx1keKztY
##
import pyautogui
from time import sleep

##
# пауза и досрочное прекращение
pyautogui.PAUSE = 1.5
pyautogui.FAILSAFE = True
##
# разрешение и позиция
pyautogui.size()
pyautogui.position()
##
# перемещение мыши
pyautogui.moveTo(1920 / 2, 1080 / 2, duration=1)
pyautogui.move(-200, -200, duration=1)
##
# нажатие
pyautogui.click()
pyautogui.doubleClick()
pyautogui.tripleClick()
pyautogui.rightClick()
pyautogui.vscroll(200)
pyautogui.middleClick()
##
# перемещение с зажатием
# pyautogui.position()
pyautogui.moveTo(491, 412, duration=1)
pyautogui.dragTo(125, 412, duration=1)
pyautogui.move(100, None, duration=1)
##
# ввод с клавиатуры
sleep(0.5)
pyautogui.typewrite('Hello, World!', interval=0.2)
##
# нажатие клавиш: press, hotkey
sleep(0.5)
pyautogui.press('enter')
pyautogui.hotkey('ctrl', 'a')
##
# скриншоты и нахождение отдельных элементов
pyautogui.screenshot(r"C:\Users\Артём Владимирович\Downloads\example.png")
pyautogui.locateCenterOnScreen(r"C:\Users\Артём Владимирович\Downloads\text.png")
pyautogui.click()







##
sleep(1)
print(pyautogui.alert(text='Давайте я покажу вам как работает эта программа', title='interceptor', button='OK'))
sleep(0.5)
print(pyautogui.confirm(text='Еще одно бесполезное окошко', title='Опрос граждан', buttons=['OK', 'Cancel']))
pyautogui.moveTo(x=1265, y=1056, duration=0.8)
pyautogui.click()
##
program_list = pyautogui.getAllTitles()
program_list
##
for i in range(40):
    if 'Новая вкладка - Google Chrome' in program_list:
        break
    sleep(0.5)
print(i)
pyautogui.moveTo(408, 57, duration=1)
pyautogui.click()
pyautogui.typewrite('spongebob cooking', interval=0.2)
pyautogui.press('enter')
pyautogui.moveTo(349, 208, duration=0.5)
pyautogui.click()
pyautogui.moveTo(1060, 418, duration=0.7)
pyautogui.click()
pyautogui.moveTo(1535, 390, duration=0.7)
pyautogui.rightClick()
pyautogui.moveTo(1550, 650, duration=0.7)
pyautogui.click()
pyautogui.moveTo(262, 1053, duration=1)
pyautogui.click()
program_list = pyautogui.getAllTitles()
for i in range(40):
    if 'Безымянный - Paint' in program_list:
        break
    sleep(0.5)
print(i)
pyautogui.hotkey('ctrl', 'v')
pyautogui.moveTo(273, 387,duration=0.5)
pyautogui.dragTo(330, 542, 1, pyautogui.easeInOutQuad)
x, y = pyautogui.locateCenterOnScreen(r"C:\Users\Артём Владимирович\Downloads\text.png")
pyautogui.click(x, y)
pyautogui.moveTo(86, 205, duration=0.5)
pyautogui.dragTo(598, 305, 0.5, pyautogui.easeInOutQuad)
pyautogui.press('capslock')
pyautogui.hotkey('shift', 'alt')
pyautogui.click()
pyautogui.typewrite("rjulf ujnjdbim vfvt tt k.,bvst rhf,c,ehuths",
                    # when you are cooking crab's burgers for your beloved mom
                    interval=0.1)  # когда готовишь маме ее любимые крабсбургеры
pyautogui.hotkey('ctrl', 'a')
pyautogui.moveTo(217, 117, duration=0.5)
pyautogui.click()
sleep(5)
pyautogui.moveTo(195, 314, duration=0.5)
pyautogui.click()
pyautogui.move(250, 50)
pyautogui.click()

##
print(pyautogui.password(text='Password l o l', title='Password', default='StrongPass', mask='*'))