#date: 2023-12-15T16:41:19Z
#url: https://api.github.com/gists/6d44141dc8a0a94c2a0e3d8f77aef319
#owner: https://api.github.com/users/SirSatorik

#Импорт библиотек
from tkinter import *
from speedtest import Speedtest

#Измерение скорости
def button():
    download = Speedtest().download()
    upload = Speedtest().upload()
    download_speed = round(download / (10**6), 2)
    upload_speed = round(upload / (10**6), 2)
    
    #Изменение текста
    download_lable.config(text='Download speed:\n' + str(download_speed) + 'MbPs')
    upload_lable.config(text='Upload speed:\n' + str(upload_speed) + 'MbPs')

#Основная часть
root = Tk()

root.title('SpeedTester')
root.geometry('300x400')

#Создание кнопки
button = Button(root, text='Start', font=40, command=button)
button.pack(side=BOTTOM, pady=40)

#Создание текста
download_lable = Label(root, text='Download speed:\n-', font=35)
download_lable.pack(pady=(50, 0))
upload_lable = Label(root, text='Upload speed:\n-', font=35)
upload_lable.pack(pady=(10, 0))

root.mainloop()
