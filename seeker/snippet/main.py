#date: 2023-11-17T16:38:16Z
#url: https://api.github.com/gists/1ac4cd979b12ca3d075674432725bd04
#owner: https://api.github.com/users/serk4s

import telebot
from telebot import types  # для указание типов
import requests
import config
import numpy as np  #импорт библиотек для работы с exlel файлов
import pandas as pd

bot = telebot.TeleBot('6498411034:AAEmtZA5dhLkHvpnJFhgq6lgVTffVmJQnUg')


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Расписание")
    btn2 = types.KeyboardButton("Погода")
    btn3 = types.KeyboardButton("Заметки")
    markup.add(btn1 , btn2, btn3)
    bot.send_message(message.chat.id ,
                     text="Привет, {0.first_name}! Я тестовый бот цифрик".format(
                         message.from_user) , reply_markup=markup)

sch = pd.read_excel("C:\cifrick_tg_bot\расписание.xlsx")
txt = str(sch.head())

@bot.message_handler(content_types=['text'])
def func(message):
    if (message.text == "Погода"):
        url = 'https://api.openweathermap.org/data/2.5/weather?q=kemerovo&units=metric&lang=ru&appid=79d1ca96933b0328e1c7e3e7a26cb347'

        weather_data = requests.get(url).json()
        print(weather_data)

        temperature = round(weather_data['main']['temp'])
        temperature_feels = round(weather_data['main']['feels_like'])

        w_now = 'Сейчас в городе Кемерово ' + ' ' + str(temperature) + ' °C'
        w_feels = 'Ощущается как ' + str(temperature_feels) + ' °C'

        bot.send_message(message.from_user.id , w_now)
        bot.send_message(message.from_user.id , w_feels)

        wind_speed = round(weather_data['wind']['speed'])
        if wind_speed < 5:
            bot.send_message(message.from_user.id , '✅ Погода хорошая, ветра почти нет')
        elif wind_speed < 10:
            bot.send_message(message.from_user.id , '🤔 На улице ветрено, оденьтесь чуть теплее')
        elif wind_speed < 20:
            bot.send_message(message.from_user.id , '❗️ Ветер очень сильный, будьте осторожны, выходя из дома')
        else:
            bot.send_message(message.from_user.id , '❌ На улице шторм, на улицу лучше не выходить')

    elif (message.text == "Расписание"):
        bot.send_message(message.chat.id, txt)

    elif (message.text == "Заметки"):
        bot.send_message(message.chat.id , "У меня нет этой функции..")

    else:
        bot.send_message(message.chat.id , text="На такую комманду я не запрограммирован...")


bot.polling(none_stop=True)
