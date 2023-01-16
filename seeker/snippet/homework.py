#date: 2023-01-16T16:44:25Z
#url: https://api.github.com/gists/18f72bc8a180acccb4242fe27e4cf852
#owner: https://api.github.com/users/Angelina91

import logging
import exceptions
import sys
import os
import requests

import telegram
import time
from telegram import Bot
from telegram import ReplyKeyboardMarkup
from telegram.ext import CommandHandler, Updater

from dotenv import load_dotenv
from http import HTTPStatus

load_dotenv()


PRACTICUM_TOKEN = "**********"
TELEGRAM_TOKEN = "**********"

TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

RETRY_PERIOD = 600
ENDPOINT = 'https://practicum.yandex.ru/api/user_api/homework_statuses/'
HEADERS = {'Authorization': "**********"


HOMEWORK_VERDICTS = {
    'approved': 'Работа проверена: ревьюеру всё понравилось. Ура!',
    'reviewing': 'Работа взята на проверку ревьюером.',
    'rejected': 'Работа проверена: у ревьюера есть замечания.'
}


 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"h "**********"e "**********"c "**********"k "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********") "**********": "**********"
    """Проверка доступности переменных окружения"""
    logging.info('Проверка доступности переменных окружения')
    return all([PRACTICUM_TOKEN, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID])


def send_message(bot, message):
    """Отправка сообщения в Telegram чат"""
    try:
        logging.info("Отправка запроса статуса")
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as error:
        logging.error(f"Ошибка отправки запроса статуса: {error}")
    else:
        logging.info("Запрос отправлен")


def get_api_answer(timestamp):
    """Запрос к единственному эндпоинту"""
    tel_timestamp = timestamp or int(time.time())
    params_request = {
        'url': ENDPOINT,
        'headers': HEADERS,
        'params': {'from_date': tel_timestamp},
    }
    message = (
        "Начал выполняться запрос: {url}, {headers}, {params}."
    ).format(**params_request)
    logging.info(message)
    try:
        response = requests.get(**params_request)
        if response.status_code != HTTPStatus.OK:
            raise exceptions.ResponseCodeError(
                f"{response.status_code} Проблема с доступом к странице"
            )
        return response.json()
    except Exception:
        raise exceptions.AvaliablePageError(
            "Неверный код ответа на запрос"
        )

def check_response(response):
    """Проверка ответа API на соответствие документации"""
    if not isinstance(response, dict):
        raise TypeError("Переменная не соответствует типу 'dict'")
    homeworks = response['homeworks']
    if not isinstance(homeworks, list):
        raise TypeError("Тип преченя домашних работ не является списком")
    return homeworks


def parse_status(homework):
    """Статус проверки конкретной домашней работы"""
    if 'homework_name' not in homework:
        raise KeyError("Такой домашней работы нет")
    homework_name = homework.get('homework_name')
    homework_status = homework.get('status')
    if homework_status not in HOMEWORK_VERDICTS:
        raise ValueError(f"Статуса работы {homework_status} нет")
    return (
        f'Изменился статус проверки работы "{homework_name}". {verdict}'
    ).format(
        homework_name=homework_name,
        verdict=HOMEWORK_VERDICTS[homework_status]
    )


def main():
    """Основная логика работы бота."""
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"h "**********"e "**********"c "**********"k "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********") "**********": "**********"
        error_text = "Всех необходимых параметров нет!Так работать не буду!"
        logging.critical(error_text)
        sys.exit(error_text)
    bot = "**********"=TELEGRAM_TOKEN)
    timestamp = int(time.time())
    start_message = "Бот готов проверять, как дела у домашек"
    send_message(bot, start_message)
    logging.info(start_message)
    prev_msg = ''

    while True:
        try:
            response = get_api_answer(timestamp)
            timestamp = response.get(
                'current_date', int(time.time())
            )
            homeworks = check_response(response)
            if homeworks:
                message = parse_status(homeworks[0])
            else:
                message = "Ничего нового не произошло"
            if message != prev_msg:
                send_message(bot, message)
                prev_msg = message
            else:
                logging.info(message)
        except Exception as error:
            message = f'Сбой в работе программы: {error}'
            logging.error(message, exc_info=True)
            if message!= prev_msg:
                send_message(bot, message)
                prev_msg = message
        finally:
            time.sleep(RETRY_PERIOD)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format=(
            '%(asctime)s, %(levelname)s, Путь - %(pathname)s, '
            'Файл - %(filename)s, Функция - %(funcName)s, '
            'Номер строки - %(lineno)d, %(message)s'
        ),
        handlers=[
            logging.FileHandler('log.txt', encoding='UTF-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    main()
