#date: 2023-01-11T17:13:56Z
#url: https://api.github.com/gists/6a11297a86baa8c4649be1b11e9401e2
#owner: https://api.github.com/users/SergeyKholopkin

import telebot
import requests
import json
TOKEN='5910168251: "**********"
bot= "**********"
list={
    'биткоин':'BTC',
    'эфириум':'ETH',
    'доллар':'USD',
}
class ConversionException(Exception):
    pass
@bot.message_handler(commands=['start', 'help'])
def help(message: telebot.types.Message):
    text='Чтобы начать работу введите команду боту в следующем формате: \n<имя валюты> \
<в какую валюту перевести>\
<количество переводимой валюты>\n<Увидеть список всех доступных валют: /values'
    bot.reply_to(message,text)
@bot.message_handler(commands=['values'])
def values(message: telebot.types.Message):
    text='Доступные валюты:'
    for a in list.keys():
        text='\n'.join((text,a,))
    bot.reply_to (message,text)

@bot.message_handler(content_types=['text',])
def convert(message: telebot.types.Message):
    values=message.text.split(' ')
    if len(values)>3:
        raise ConversionException('слишком много параметров')

    quote, base, amount=values
    if quote==base:
        raise ConversionException(f'Невозможно перевести одинаковые валюты{base}')
    try:
        quote_ticker=list[quote]#проверка соответствия вводимого значения словарю
    except KeyError:
        raise ConversionException(f'Не удалось обработать валюту{quote}')
    try:
        base_ticker=list[base]#проверка соответствия вводимого значения словарю
    except KeyError:
        raise ConversionException(f'Не удалось обработать валюту{base}')
    try:
        amount=float(amount)
    except ValueError:
        raise ConversionException(f'Не удалось обработать количество{amount}')#проверка соответствия кол-ва

    # quote_ticker, base_ticker=list[quote],list[base]
    r = requests.get(f'https://min-api.cryptocompare.com/data/price?fsym={quote_ticker}&tsyms={base_ticker}')
    total_base=json.loads(r.content)[list[base]]
    text=f'Цена{amount}{quote} в {base}-{total_base}'
    bot.send_message(message.chat.id,text)


# @bot.message_handler()
# def echo_test(message: telebot.types.Message):
#     bot.send_message(message.chat.id,'Hello')


bot.polling()chat.id,'Hello')


bot.polling()