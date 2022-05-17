#date: 2022-05-17T17:17:32Z
#url: https://api.github.com/gists/6ff1be9a48714781056e9c6cfb995623
#owner: https://api.github.com/users/Sparrowkill

import telebot
from telebot import types
from magicballon1 import *
import time
from random import choice

token = '5351198934:AAFeKLQjtcetjlr6m_7QQdITg29jkklBHXI'
bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start', 'reset'])
def message(message):
    bot.send_message(message.chat.id, answer1)
    time.sleep(1)
    sent = bot.send_message(message.from_user.id, name)
    bot.register_next_step_handler(sent, UserName)


def UserName(message):
    UserName_message = message.text
    bot.send_message(message.from_user.id, f'Я буду рад помочь тебе {UserName_message} с твоим вопросом')
    time.sleep(1)
    question = bot.send_message(message.from_user.id, question_user)
    bot.register_next_step_handler(question, answer_question)


@bot.message_handler(func=lambda n: n.text == 'Да')
@bot.message_handler(func=lambda m: m.text == 'Задать вопрос снова')
def yes(message):
    question = bot.send_message(message.from_user.id, question_user)
    bot.register_next_step_handler(question, answer_question)


def answer_question(message):
    Board = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1, one_time_keyboard=True)
    Yes = types.KeyboardButton(text='Да')
    No = types.KeyboardButton(text='Нет')
    Board.add(Yes, No)
    bot.send_message(message.from_user.id, choice(answer))
    time.sleep(1)
    bot.send_message(message.from_user.id, answer_user, reply_markup=Board)


@bot.message_handler(func=lambda m: m.text == 'Нет')
def no_answer(message):
    board = types.ReplyKeyboardMarkup(resize_keyboard=True)
    question = types.KeyboardButton(text='Задать вопрос снова')
    board.add(question)
    bot.send_message(message.from_user.id, good_bye, reply_markup=board)


bot.polling()
