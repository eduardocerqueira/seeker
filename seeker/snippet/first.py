#date: 2022-06-07T16:58:08Z
#url: https://api.github.com/gists/9ac813025fa515bb3f4aa59951ee354b
#owner: https://api.github.com/users/kkamagwi

import telebot


TOKEN = 'token'
bot = telebot.TeleBot(TOKEN, parse_mode='HTML') 


@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    item1 = '10.00'
    item2 = '20.00'
    user_markup.add(item1, item2)

    bot.reply_to(message, f'Hello {message.from_user.first_name}, when should I track your time?', reply_markup=user_markup)


bot.infinity_polling()