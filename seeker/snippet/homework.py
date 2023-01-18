#date: 2023-01-18T16:40:52Z
#url: https://api.github.com/gists/11c70a58ac6eee89b54554466f352fd6
#owner: https://api.github.com/users/Angelina91

import logging
import exceptions
import sys
import os
import requests

import telegram
import time

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
    logging.debug('Проверка доступности переменных окружения')
    return all([PRACTICUM_TOKEN, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID])


def send_message(bot, message):
    """Отправка сообщения в Telegram чат"""
    try:
        logging.debug("Отправка запроса статуса")
        bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )
        logging.debug("На старт!Сообщение отправлено")
    except Exception as error:
        logging.error(f"Ошибка - {error}")
    # except Exception as error:
    #     raise exceptions.SendMessageException(error)


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
    logging.debug(message)
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
    try:
        verdict = HOMEWORK_VERDICTS[homework.get('status')]
        homework_name = homework['homework_name']
    except Exception as error:
        logging.error(f"Ошибка при запросе - {error}")
    return (
        f'Изменился статус проверки работы "{homework_name}". {verdict}'
    )


def main():
    """Основная логика работы бота."""
    _logging_format = '%(asctime)s, %(levelname)s, %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=_logging_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"h "**********"e "**********"c "**********"k "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********") "**********": "**********"
        error_text = "Всех необходимых параметров нет!Так работать не буду!"
        logging.critical(error_text)
        sys.exit(error_text)
    

    bot = "**********"=TELEGRAM_TOKEN)
    logging.debug('Бот запущен')
    timestamp = ''

    while True:
        try:
            response = get_api_answer(timestamp)
            homeworks = check_response(response)
            homework = homeworks[0]
            message = parse_status(homework)
            send_message(bot, message)
            timestamp = response.get('current_date')
        except IndexError:
            logging.debug('Sorry!Нет никаких обновлений')
            timestamp = response.get('current_date')
        except TypeError as error:
            message = f'Тип данных не тот: {error}'
            logging.error(message)
        except KeyError as error:
            message = f'Ключевая ошибка: {error}'
            logging.error(message)
        except exceptions.SendMessageException as error:
            message = f'Не удалось отправить сообщение в Telegram - {error}'
            logging.error(message)
        except exceptions.AvaliablePageError as error:
            message = f'ENDPOINT недоступен. Код ответа API: {error}'
            logging.error(message)
        except Exception as error:
            message = f'Сбой в работе программы: {error}'
            logging.error(message)
        finally:
            time.sleep(RETRY_PERIOD)


if __name__ == '__main__':
    main()   
