#date: 2022-05-17T17:01:05Z
#url: https://api.github.com/gists/680f08c27cdd662c9c793ca5e5f079bc
#owner: https://api.github.com/users/PuffyWithEyes

import itertools
import requests
from bs4 import BeautifulSoup
import fake_useragent
from os import system
from sys import platform
from datetime import datetime


class Word:
    def __init__(self):
        if platform == 'win32':
            system('cls')
        elif platform == 'linux':
            system('clear')
        else:
            print('[INFO] This soft only for users of Windows and Linux')
            exit(0)

        useragent = fake_useragent.UserAgent().random
        self.headers = {
            'User-Agent': useragent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
        self.main_list = []

        self._ask_info()

    def _find_word(self, data: str, count: int, counter=0, percent=0):
        num = len(data) ** int(count)
        print(f"[INFO] Поиск начался! Будет проверено {num} слов.")
        if num // 100 != 0 or num // 100 != 1:
            num = num // 100
        else:
            num = 1

        for i in itertools.product(data, repeat=count):
            start = ''
            for k in i:
                start += str(k)

            url = f'http://gramota.ru/slovari/dic/?lop=x&bts=x&zar=x&ag=x&ab=x&sin=x&lv=x&az=x&pe=x&word=' \
                  f'{start.lower()}'
            r = requests.get(url=url, headers=self.headers)
            bs = BeautifulSoup(r.text, 'lxml')
            try:
                find = bs.find('div', class_='inside block-content').find('div', style='padding-left:50px').text
            except:
                find = 'искомое слово отсутствует'
            counter += 1

            if find != 'искомое слово отсутствует':
                print("[INFO] НАЙДЕНО СЛОВО:", start)
                self.main_list.append(start)

            if counter % num == 0 or percent == 0 and num > 1:
                print(f"[INFO: {datetime.now().time()}; Выполнено: {percent}%] Проверено {counter} слов")
                percent += 1
            elif counter % num == 0 or percent == 0 and num == 1:
                print(f"[INFO: {datetime.now().time()}] Проверено {counter} слов")
                percent += 1

    def _ask_info(self):
        data = input('Введите все буквы русского алфавита, комбинацию из которых вам нужно найти: ')
        while True:
            try:
                count = int(input('Введите кол-во символов в слове: '))
                try:
                    self._find_word(data=data, count=int(count))
                except Exception as ex:
                    print(ex)
                    exit(-1)

                if self.main_list:
                    print('[INFO] ВСЕ НАЙДЕННЫЕ СЛОВА:')
                    for i in self.main_list:
                        print(i)
                break
            except:
                print('[INFO] Вы ввели не целое число!')


if __name__ == '__main__':
    Word()
