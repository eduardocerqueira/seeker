#date: 2024-03-12T16:42:57Z
#url: https://api.github.com/gists/7b3ef28f6e14602a93fac9c3cf28a9c2
#owner: https://api.github.com/users/andmerk93

'''
Скрипт на Selenium, заходит на https://www.nseindia.com/
и имитирует пользовательское поведение:
1. Зайти на главную страницу
2. Пролистать вниз до графика
3. Выбрать график "NIFTY BANK"
4. Нажать “View all” под "TOP 5 STOCKS - NIFTY BANK"
5. Выбрать в селекторе “NIFTY ALPHA 50”
6. Пролистать таблицу до конца

Основная логика в функции nseindia_user_behavior.

Программа запускает существующую версию firefox,
пристегивает к ней geckodriver, а к нему Selenium.
При падении завершает процессы, и выводит ошибку в консоль.

Сделано в качестве тестового задания в марте 2024.
'''

from subprocess import Popen
import time

from selenium.webdriver.common.by import By
from selenium.webdriver import FirefoxOptions, Keys, Remote
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import ElementNotInteractableException


def nseindia_user_behavior(driver):
    # главная страница
    driver.get('https://www.nseindia.com/')
    # пользователь думает
    time.sleep(2)
    # выбор графика NIFTY BANK
    bank_label = driver.find_element(By.CSS_SELECTOR, 'a#tabList_NIFTYBANK')
    bank_label.click()
    # пролистать до графика
    bank_label.send_keys(Keys.PAGE_DOWN)
    time.sleep(3)
    # Посмотреть всё
    driver.find_element(
        By.CSS_SELECTOR, 'div#tab4_gainers_loosers'
    ).find_element(By.CSS_SELECTOR, 'span#viewall').click()
    # пользователь думает
    time.sleep(2)
    # Выбор NIFTY 50 из выпадающего меню
    Select(driver.find_element(By.CSS_SELECTOR, 'select#equitieStockSelect')).select_by_value('NIFTY 50')
    time.sleep(7)
    # Листаем таблицу до конца
    driver.find_element(By.CSS_SELECTOR, 'table#equityStockTable').send_keys(Keys.ARROW_DOWN)


def runner():
    firefox = Popen(
        ('C:\\Program Files\\Mozilla Firefox\\firefox.exe', '-start-debugger-server', '2828', '-marionette')
    )
    geckodriver = Popen(
        ('D:\\python\\geckodriver.exe', '--connect-existing', '--marionette-port', '2828')
    )
    driver = Remote(
        command_executor='http://127.0.0.1:4444',
        options=FirefoxOptions()
    )
    try:
        nseindia_user_behavior(driver)
    except ElementNotInteractableException:
        time.sleep(5)
        print('Bottom of table is reached')
        # Селениум споткнулся
        # об конец таблицы,
        # ожидаемое поведение
    except Exception as exc:
        print('Unexpected error')
        print(exc)
        # Отлов всех остальных ошибок
        # с выводом в консоль
    geckodriver.terminate()
    firefox.terminate()


if __name__ == '__main__':
    runner()
