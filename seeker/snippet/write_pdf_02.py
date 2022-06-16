#date: 2022-06-16T17:08:09Z
#url: https://api.github.com/gists/9d277b84d919a35329051ede6a38bb21
#owner: https://api.github.com/users/SoftSAR

#Импортируем библиотеку
from fpdf import FPDF
#Задаем параметры страницы и единицы измерения
pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.add_page()#Создаем страницу
#Добавляем шрифт с поддержкой UNICOD
pdf.add_font('DejaVu', fname='font\\DejaVuSansCondensed.ttf')
pdf.set_font('DejaVu', size=14)#Устанавливаем параметры ширфта

text = """
English: Hello World
Greek: Γειά σου κόσμος
Polish: Witaj świecie
Portuguese: Olá mundo
Russian: Здравствуй, Мир
Vietnamese: Xin chào thế giới
Arabic: مرحبا العالم
Hebrew: שלום עולם
"""

for txt in text.split('\n'):
    pdf.write(8, txt)
    pdf.ln(8)
pdf.output("Example.pdf")