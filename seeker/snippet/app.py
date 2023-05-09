#date: 2023-05-09T16:44:07Z
#url: https://api.github.com/gists/d8b7d43e70cdcbfb85d1e82454988403
#owner: https://api.github.com/users/Mad-Max1

import telebot
from config import keys, TOKEN
from extensions import ConvertException, CryptoConverter

bot = "**********"

@bot.message_handler(commands=['start', 'help'])
def help(message: telebot.types.Message):
    text = 'Чтобы начать работу введите команду боту в следующем формате:\n<имя валюты цену которой хотите узнать><пробел> \
<имя валюты в которой надо узнать цену первой валюты><пробел> \
<количество первой валюты(только целые значения)>\nУвидеть список всех доступных валют: /values'
    bot.reply_to(message, text)

@bot.message_handler(commands=['values'])
def values(message: telebot.types.Message):
    text = 'Доступные валюты:'
    for key in keys.keys():
        text = '\n'.join((text, key, ))
    bot.reply_to(message, text)

@bot.message_handler(content_types=['text', ])
def convert(message: telebot.types.Message):
    try:
        values = message.text.split(' ')

        if len(values) != 3:
            raise ConvertException("Неверное кол-во параметров.")

        base, quote, amount = values
        total_base = CryptoConverter.get_price(base, quote, amount)
    except ConvertException as e:
        bot.reply_to(message, f'Ошибка пользователя\n{e}')
    except Exception as e:
        bot.reply_to(message, f'Не удалось обработать команду\n{e}')
    else:
        text = f'Цена {amount} {base} в {quote} - {total_base}'
        bot.send_message(message.chat.id, text)

bot.polling().polling()