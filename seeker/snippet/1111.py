#date: 2022-07-21T16:54:29Z
#url: https://api.github.com/gists/5766f8d10dd0ff196ebaafa365f3e80b
#owner: https://api.github.com/users/istiz

import telebot, random

bot = telebot.TeleBot('5579719476:AAFFEbeNzVF8SAjX8YQcK6JXruwmKFHBsq8')

@bot.message_handler(content_types=('text'))

def game(message):
    if message.text == 'Го':
        bot.send_message(message.from_user.id, 'Привет!Сыграем в игру?')
        bot.register_next_step_handler(message, start)
    else:
        bot.send_message(message.from_user.id, 'Напиши "Го"')

def start(message):
    bot.send_message(message.from_user.id,'Запускаем игру?')

    VIS = (
        """
        П
        """,
        """
        П
        И
        """,
        """
        П
        И
        С
        """,
        """
        П
        И
        С
        Е
        """,
        """
        П
        И
        С
        Е
        Ц
        """
    )

    max_wrong = len(VIS)
    words = ('мазута','табуретка','сигареты')

    word = random.choice(words)
    so_far = '_'*len(word)
    wrong = 0
    used = []

    while wrong < max_wrong and so_far != word:
        bot.send_message(message.from_user.id,VIS[wrong])



        guess =  bot.send_message(message.from_user.id,'\nВведите свое предположение:')

        while guess in used:
            bot.send_message(message.from_user.id,'Вы уже уналаои букву', guess)
            guess = bot.send_message(message.from_user.id,'\nВведите свое предположение:')

        used.append(guess)

        if guess in word:
            bot.send_message(message.from_user.id,'\nДа!', guess, "есть в слове")

            new =''
            for i in range(len(word)):
                if guess == word[i]:
                    new += guess
                else:
                    new += so_far[i]
            so_far = new
        else:
            bot.send_message(message.from_user.id,'\nИзвините, буквы \'' + guess + '\' нет в слове.')
            wrong += 1
    if wrong == max_wrong:
        bot.send_message(message.from_user.id,VIS[wrong])
        bot.send_message(message.from_user.id,'\nТебе ПЕЗДА')
    else:
        bot.send_message(message.from_user.id,'\nВы угадали слово!')

    bot.send_message(message.from_user.id,'\nЗанаданное слово было \'' + word + '\'')


bot.polling(non_stop=True, interval=0)




