#date: 2024-07-18T16:51:56Z
#url: https://api.github.com/gists/734f884a9f40300b14b102eb20854bcc
#owner: https://api.github.com/users/BorisITZaitsev

import telebot
from custom_library import *


bot = telebot.TeleBot("6621812434:AAGp48AQa2sGh8n0Bljpi4sOJUi-0b_IYrc")


@bot.message_handler(commands=["clear_speakers"])
def clear_speakers(message):
    try:
        clear(message.chat.id)
        bot.send_message(message.chat.id, "Готово.")
        menu(message)
    except (KeyError, ValueError, FileNotFoundError):
        bot.send_message(message.chat.id, "Возникли проблемы с удалением. Операция отклонена.")
        menu(message)


@bot.message_handler(commands=["students"])
def students(message):
    names = get_full_data(message.chat.id)
    coming_text = ""
    for i in names:
        coming_text += f"'{i}'\n"
    bot.send_message(message.chat.id, coming_text)
    menu(message)


@bot.message_handler(commands=["delete"])
def delete_theme0(message):
    if hm_themes(message.chat.id) == 0:
        bot.send_message(message.chat.id, "Пока нечего удалять. Добавьте темы.")
        menu(message)
    else:
        msg = bot.send_message(message.chat.id, "ФИО студента, тему которого нужно удалить.",
                               reply_markup=telebot.types.ReplyKeyboardRemove())
        bot.register_next_step_handler(msg, delete_theme1)


def delete_theme1(message):
    try:
        student = message.text
        themes = get_full_data(message.chat.id)[student]
        markup = telebot.types.ReplyKeyboardMarkup()
        if len(themes) == 1:
            bot.send_message(message.chat.id, "У этого студента нечего удалять. Добавьте темы.")
            menu(message)
        else:
            for i in range(1, len(themes)):
                markup.add(telebot.types.KeyboardButton(str(i)))
            coming_text = "Номер темы, которую нужно удалить."
            msg = (bot.send_message(message.chat.id, text=coming_text.format(message.from_user), reply_markup=markup))
            bot.register_next_step_handler(msg, delete_theme2, student)
    except KeyError:
        bot.send_message(message.chat.id, "Появились проблемы с нахождением имени студента. Операция невозможна")
        menu(message)


def delete_theme2(message, student_name):
    theme_number = int(message.text)
    try:
        theme = theme_remove(message.chat.id, student_name, theme_number)
        coming_text = f"Тема: '{theme}' у студента - {student_name} успешно удалена."
        bot.send_message(message.chat.id, coming_text)
        menu(message)
    except KeyError:
        bot.send_message(message.chat.id, "Возникли проблемы с удалением. Операция отклонена.")
        menu(message)


@bot.message_handler(commands=['set_by_hand'])
def set_by_hand0(message):
    try:
        further_speakers(message.chat.id)
        msg = bot.send_message(message.chat.id, "Пожалуйста, введите изменения. Пример:\n"
                                                "Иванов Иван Иванович - Тема1\n"
                                                "Александров Александр Александрович - Тема2")
        bot.register_next_step_handler(msg, set_by_hand1)
    except FileNotFoundError:
        bot.send_message(message.chat.id, "Боюсь, процесс регистрации был пройжен некорретно. Введите "
                                          "стартер заново.")


def set_by_hand1(message):
    further_speakers(message.chat.id)
    changes = str(message.text).split("\n")
    for i in range(0, len(changes)):
        student = changes[i].split(" - ")[0]
        try:
            single_apply(changes[i], str(message.chat.id))
        except KeyError:
            bot.send_message(message.chat.id, f"Нет такого студента, как {student}."
                                              f" Придётся полностью произвести операцию присваивания заново.")
        except IndexError:
            bot.send_message(message.chat.id, "Неправильный формат ввода. Повторите операцию заново.")
        except FileNotFoundError:
            bot.send_message(message.chat.id, "Боюсь, процесс регистрации был пройжен некорретно. Введите "
                                              "стартер заново.")
    menu(message)


@bot.message_handler(commands=['get_full_base'])
def send_full_data(message):
    try:
        data = get_full_data(message.chat.id)
        coming_text = ""
        for i in data:
            if data[i][0] == 0:
                coming_text += f"{i}. Всего брал(a) тем: {data[i][0]}.\n\n"
            else:
                coming_text += f"{i}. Всего брал(a) тем: {data[i][0]}. Темы:"
                for j in data[i][1:]:
                    coming_text += ' "' + j + '" '
                coming_text += "\n\n"
        bot.send_message(message.chat.id, coming_text)
        menu(message)
    except FileNotFoundError:
        bot.send_message(message.chat.id, "Нет такого файла. Прошу пройти инициализацию заново.")


@bot.message_handler(commands=['get_further_speakers'])
def send_speakers(message):
    coming_text = ""
    try:
        todays_speakers = further_speakers(message.chat.id)
        for i in todays_speakers:
            coming_text += f"{i} - {todays_speakers[i]}\n\n"
        bot.send_message(message.chat.id, coming_text)
        menu(message)
    except (telebot.apihelper.ApiException, FileNotFoundError):
        bot.send_message(message.chat.id, "Данный список пока пуст или инициализация пройдена неправильно.")
        menu(message)


@bot.message_handler(content_types=["document"])
def handle_file(message):
    chat_id = message.chat.id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    try:
        if message.document.file_name.split('.')[-1] != "docx":
            bot.send_message(chat_id, "Ты перепутал формат. Мне нужен DOCX!!!",
                             reply_markup=telebot.types.ReplyKeyboardRemove())
        else:
            if "тем" in message.document.file_name.lower():
                src = f'C:/Users/borya/PycharmProjects/pythonProject/pythonbot_mgimo/{chat_id}/th_list.docx'
                with open(src, 'wb') as new_file:
                    new_file.write(downloaded_file)
                connector(chat_id)
                bot.reply_to(message, "Супер. Раздача тем выполнена успешно.\n")
            else:
                folder(str(chat_id))
                src = f'C:/Users/borya/PycharmProjects/pythonProject/pythonbot_mgimo/{chat_id}/st_list.docx'
                with open(src, 'wb') as new_file:
                    new_file.write(downloaded_file)
                bot.reply_to(message, "Спасибо. Создаю базу данных на вашу группу.")
                database_create(str(chat_id))
                bot.send_message(chat_id, "База данных успешно создана.")
            coming_text = "Ожидаю дальнейших команд или файлов с темами."
            bot.send_message(chat_id, coming_text)
            menu(message)
    except (AttributeError, KeyError, ValueError, FileNotFoundError):
        bot.send_message(chat_id, "Что-то пошло не так. Возвращаю вас в главное меню.")
        menu(message)


@bot.message_handler(commands=['start', 'menu'])
def menu(message):
    if not base_existence(message.chat.id):
        bot.send_message(message.chat.id, "Для начала пришлите список группы в формате '.docx'.", )

    else:
        coming_text = "Обратите внимание на кнопки под клавиатурой. Вам доступны некоторые запросы."
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = telebot.types.KeyboardButton("Полная база")
        btn2 = telebot.types.KeyboardButton("След. доклады")
        btn3 = telebot.types.KeyboardButton("Задать вручную")
        btn4 = telebot.types.KeyboardButton("Удалить тему")
        btn5 = telebot.types.KeyboardButton("ФИО студентов")
        btn6 = telebot.types.KeyboardButton("Темы сданы")
        markup.add(btn1, btn2, btn3, btn4, btn5, btn6)
        bot.send_message(message.chat.id, text=coming_text.format(message.from_user), reply_markup=markup)


@bot.message_handler(content_types=['text'])
def text_handler(message):
    sense = message.text
    if sense == "Меню":
        menu(message)
    if sense == "След. доклады":
        send_speakers(message)
    if sense == "Задать вручную":
        set_by_hand0(message)
    if sense == "Полная база":
        send_full_data(message)
    if sense == "Удалить тему":
        delete_theme0(message)
    if sense == "ФИО студентов":
        students(message)
    if sense == "Темы сданы":
        clear_speakers(message)


bot.polling(none_stop=True)
