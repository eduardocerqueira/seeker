#date: 2024-11-15T17:05:54Z
#url: https://api.github.com/gists/7732c7ab7a1ac7517bac4c7a09fffb32
#owner: https://api.github.com/users/pesokxx

import telebot
from telebot import types
import sqlite3

# Токен вашего бота
TOKEN = "**********"

# Создаем объект бота
bot = "**********"


# Подключение к базе данных
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn


# Создание таблицы, если она еще не существует
def create_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            city TEXT,
            interests TEXT,
            goal TEXT
        )
    ''')
    conn.commit()


@bot.message_handler(commands=['start'])
def start(message):
    with get_db_connection() as conn:
        create_table(conn)
        # Отправляем приветственное сообщение
        bot.send_message(
            message.chat.id,
            "Привет! Давай заполним твою анкету.\n\nКак тебя зовут?"
        )

        # Сбрасываем состояние текущего пользователя
        conn.execute("DELETE FROM users WHERE user_id=?", (message.from_user.id,))
        conn.commit()

        # Устанавливаем следующий шаг обработки сообщений
        bot.register_next_step_handler(message, process_name)


def process_name(message):
    with get_db_connection() as conn:
        try:
            name = message.text.strip()

            if not name.isalpha():
                raise ValueError("Имя должно содержать только буквы.")

            conn.execute("UPDATE users SET name=? WHERE user_id=?", (name, message.from_user.id))
            conn.commit()

            bot.send_message(message.chat.id, "Отлично! Теперь напиши свой возраст.")
            bot.register_next_step_handler(message, process_age)
        except Exception as e:
            bot.reply_to(message, f"Произошла ошибка при обработке имени: {e}")


def process_age(message):
    with get_db_connection() as conn:
        try:
            age = int(message.text.strip())

            if age < 18 or age > 100:
                raise ValueError("Возраст должен быть между 18 и 100 годами.")

            conn.execute("UPDATE users SET age=? WHERE user_id=?", (age, message.from_user.id))
            conn.commit()

            bot.send_message(message.chat.id, "Теперь укажи город проживания.")
            bot.register_next_step_handler(message, process_city)
        except ValueError as e:
            bot.reply_to(message, f"Пожалуйста, введите правильный возраст: {e}")


def process_city(message):
    with get_db_connection() as conn:
        city = message.text.strip().capitalize()

        if len(city) == 0:
            raise ValueError("Город не может быть пустым.")

        conn.execute("UPDATE users SET city=? WHERE user_id=?", (city, message.from_user.id))
        conn.commit()

        bot.send_message(
            message.chat.id,
            "Какие у тебя интересы? (Перечислите их через запятую)"
        )
        bot.register_next_step_handler(message, process_interests)


def process_interests(message):
    with get_db_connection() as conn:
        interests = message.text.strip().lower()

        if len(interests) == 0:
            raise ValueError("Интересы не могут быть пустыми.")

        conn.execute("UPDATE users SET interests=? WHERE user_id=?", (interests, message.from_user.id))
        conn.commit()

        keyboard = types.InlineKeyboardMarkup(row_width=2)
        button_friends = types.InlineKeyboardButton(text="Найти друзей", callback_data='friends')
        button_partner = types.InlineKeyboardButton(text="Найти пару", callback_data='partner')
        keyboard.add(button_friends, button_partner)

        bot.send_message(
            message.chat.id,
            "Выберите свою цель:",
            reply_markup=keyboard
        )


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    with get_db_connection() as conn:
        if call.data in ['friends', 'partner']:
            conn.execute(
                "UPDATE users SET goal=? WHERE user_id=?",
                (call.data, call.from_user.id)
            )
            conn.commit()

            bot.edit_message_text(
                chat_id=call.message.chat.id,
                message_id=call.message.message_id,
                text=f"Ваша цель: {call.data}. Анкета заполнена!"
            )

            show_matching_profiles(call.message)


def show_matching_profiles(message):
    with get_db_connection() as conn:
        current_user = get_current_user(conn, message.from_user.id)

        if current_user is None:
            bot.send_message(message.chat.id, "Пользователь не найден. Попробуйте снова.")
            return

        query = '''
            SELECT * FROM users
            WHERE user_id != ? AND age = ? AND city = ? AND goal = ?
            ORDER BY RANDOM() LIMIT 10
        '''

        cur = conn.cursor()
        cur.execute(query, (current_user['user_id'], current_user['age'], current_user['city'], current_user['goal']))
        rows = cur.fetchall()


        for row in rows:
            profile = f"""
            Имя: {row[1]}
            Возраст: {row[2]}
            Город: {row[3]}
            Интересы: {row[4]}
            Цель: {row[5]}
            """

        keyboard = types.InlineKeyboardMarkup(row_width=2)
        like_button = types.InlineKeyboardButton(text="👍", callback_data=f'like_{row[0]}')
        dislike_button = types.InlineKeyboardButton(text="👎", callback_data=f'dislike_{row[0]}')
        keyboard.add(like_button, dislike_button)

        bot.send_message(
            message.chat.id,
            profile,
            reply_markup=keyboard
        ) \
 \
        @ bot.callback_query_handler(func=lambda call: call.data.startswith('like_'))


def handle_like(call):
    with get_db_connection() as conn:
        user_id = int(call.data.split('_')[1])
        conn.execute("INSERT INTO matches (user_id, matched_user_id) VALUES (?, ?)", (call.from_user.id, user_id))
        conn.commit()

        bot.answer_callback_query(call.id, "Понравилось!")

        show_matching_profiles(call.message) \
 \
        @ bot.callback_query_handler(func=lambda call: call.data.startswith('dislike_'))


def handle_dislike(call):
    bot.answer_callback_query(call.id, "Упс, не понравилось...")

    show_matching_profiles(call.message)


def get_current_user(conn, user_id):
    cur = conn.cursor()  # Получение курсора
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()  # Вызываем fetchone() для курсора

    if row is None:
        return None  # Возвращаем None, если пользователь не найден

    return {
        'user_id': row[0],
        'name': row[1],
        'age': row[2],
        'city': row[3],
        'interests': row[4],
        'goal': row[5]
    }


if __name__ == '__main__':
    bot.polling(none_stop=True)op=True)