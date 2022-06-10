#date: 2022-06-10T16:59:54Z
#url: https://api.github.com/gists/a74ebed7e3f29b2757d3e631d329459d
#owner: https://api.github.com/users/SoftSAR

import cv2
import pytesseract

img = cv2.imread('test.png') #Открываем изображение
img_2RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Переводим изображение в формат RGB

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#Настройки tesseract: https://help.ubuntu.ru/wiki/tesseract#tesseract
config = r'--oem 3 --psm 6' #Настройки конфигурации oem - версия движка psm - версия формата изображения

text = pytesseract.image_to_data(img_2RGB, config=config)

# Перебираем данные текстовые надписи
for i, el in enumerate(text.splitlines()):
    if i != 0:        
        el = el.split()
        try:
	    # Выделяем текст на картинке
            x, y, w, h = int(el[6]), int(el[7]), int(el[8]), int(el[9])
            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 2)		
        except:
            pass

# Отображаем фото
cv2.imshow('Result', img)
cv2.waitKey(0)