#date: 2024-09-26T16:59:15Z
#url: https://api.github.com/gists/22c8b3c3a8232f2d990d1f9bb2b20aa8
#owner: https://api.github.com/users/OlgaZhivaeva

import argparse
import requests
import telegram
from environs import Env
from time import sleep


def main():
    env = Env()
    env.read_env()
    tg_bot_token = "**********"
    dvmn_token = "**********"
    tg_chat_id = env.str('TG_CHAT_ID')

    bot = "**********"=tg_bot_token)

    url = 'https://dvmn.org/api/long_polling/'
    headers = {'Authorization': "**********"
    params = None

    while True:
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            attempt = response.json()
        except requests.exceptions.Timeout:
            print('timeout')
            continue
        except requests.ConnectionError:
            print('ConnectionError')
            sleep(5)
            continue
        if attempt['status'] == 'timeout':
            timestamp = attempt['timestamp_to_request']
            params = {'timestamp': timestamp}
            continue

        timestamp = attempt['last_attempt_timestamp']
        params = {'timestamp': timestamp}

        lesson_title = attempt['new_attempts'][0]['lesson_title']
        is_negative = attempt['new_attempts'][0]['is_negative']
        lesson_url = attempt['new_attempts'][0]['lesson_url']

        if is_negative:
            bot.send_message(text=f"У Вас проверили работу \"{lesson_title}\".\n"
                                  f"К сожалению, в работе нашлись ошибки.\n{lesson_url}",
                             chat_id=tg_chat_id)
            continue

        bot.send_message(text=f"У Вас проверили работу \"{lesson_title}\".\n"
                              f"Преподавателю все понравилось. Можно приступать к следующему уроку!",
                         chat_id=tg_chat_id)


if __name__ == "__main__":
    main()
= "__main__":
    main()
