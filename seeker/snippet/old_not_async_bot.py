#date: 2024-05-29T16:58:21Z
#url: https://api.github.com/gists/853f74403aed8c7efe6b30e7195d71c1
#owner: https://api.github.com/users/adepte-fufayka

import telebot
import re
import time
import random
import CONFIG
print('включен')

TOKEN = "**********"
bot = "**********"
cities = [["🏛", "Александрия"], ["🏭", "Порт-Романтик"], ["🏢", "Эндимион"], ["🏣", "Китс"]]
zones = ['Безопасные земли', 'Дикие земли', 'Городской зоопарк', 'Квартал Коми-Конщиков', 'Азиатское гетто',
         'Фантастические твари']

# print(time.time())
last_update = int(time.time())
lab_kb = telebot.types.InlineKeyboardMarkup(row_width=2)
# lab_kb.add(telebot.types.InlineKeyboardButton(text='🔼 Качество', callback_data='+quality'),
#        telebot.types.InlineKeyboardButton(text='🔽 Качество', callback_data='-quality'))
lab_kb.add(telebot.types.InlineKeyboardButton(text='🔼 Тюнинг', callback_data='+tuning'),
           telebot.types.InlineKeyboardButton(text='🔽 Тюнинг', callback_data='-tuning'))
lab_kb.add(telebot.types.InlineKeyboardButton(text='🔼 Заточка', callback_data='+sharpening'),
           telebot.types.InlineKeyboardButton(text='🔽 Заточка', callback_data='-sharpening'))


class Place:
    def __init__(self, name, _type, found, zone, x, y):
        self.name = name
        self._type = _type
        self.found = found
        self.zone = zone
        self.x = x
        self.y = y

    def __str__(self):
        return f'{self.name}\n{self._type}\n{self.found}\n{self.zone}\n{self.x}\n{self.y}\n'


class User:
    def __init__(self, uid, username, name, squad_name, time, res_time, deff, attack, health_p, power_p, mana_p, role,
                 boss_ping, city, prof, prof_time, timezone):
        self.uid = uid
        self.city = city
        self.res_time = res_time
        self.deff = deff
        self.attack = attack
        self.username = username
        self.name = name
        self.squad_name = squad_name
        self.time = time
        self.mana_p = mana_p
        self.power_p = power_p
        self.health_p = health_p
        self.role = role
        self.boss_ping = boss_ping
        self.prof = prof
        self.prof_time = prof_time
        self.timezone = timezone

    def __str__(self):
        return f'{self.uid}\n{self.username}\n{self.name}\n{self.squad_name}\n{self.time}\n{self.res_time}\n{self.deff}\n{self.attack}\n{self.health_p}\n{self.power_p}\n{self.mana_p}\n{self.role}\n{self.boss_ping}\n{self.city}\n{self.prof}\n{self.prof_time}\n{self.timezone}\n'


users = []
f = open("users.txt", "r", encoding='utf-8')
s = f.readlines()
k = 17
for i in range(len(s) // k):
    users.append(User(int(s[i * k]), s[i * k + 1][:-1], s[i * k + 2][:-1], s[i * k + 3][:-1], int(s[i * k + 4][:-1]),
                      int(s[i * k + 5][:-1]), int(s[i * k + 6][:-1]), int(s[i * k + 7][:-1]), int(s[i * k + 8][:-1]),
                      int(s[i * k + 9][:-1]),
                      int(s[i * k + 10][:-1]), s[i * k + 11][:-1], False if (s[i * k + 12][:-1]) == 'False' else True,
                      s[i * k + 13][:-1],
                      s[i * k + 14][:-1], int(s[i * k + 15][:-1]), int(s[i * k + 16][:-1])))
f.close()
shmot_quality = ['Качество: ▫️ Плохое', 'Качество: ▪️ Обычное', 'Качество: 🔹 Необычное', 'Качество: 🔸 Редкое',
                 'Качество: 🔺 Эпическое']
shmot_dops = [['Шанс выпадения вещей:', 0.3], ['Вампиризм:', 0.25], ['Игнор. брони в PVP:', 0.4],
              ['Восстановление 🔮:', 0.1],
              ['Отражение урона:', 1],
              ['Увеличение скорости восстановления энергии:', -0.5],
              ['Увеличение прочности:', 5], ['Доп. 💰 с продажи вещей:', 1], ['Доп. 🌟 с мобов:', 0.25],
              ['Качество дропа:', 0.5], ['Доп. 🔮:'], ['Доп. ❤️:'], ['Доп. 💪:']]
places = []
f = open('places.txt', encoding='utf-8')
s = f.readlines()
k = 6
for i in range(len(s) // k):
    places.append(
        Place(s[i * k][:-1], int(s[i * k + 1]), False if (s[i * k + 2][:-1]) == 'False' else True, int(s[i * k + 3]),
              int(s[i * k + 4]),
              int(s[i * k + 5])))
    # print(s[i * k + 2][:-1])
f.close()


print('начинаю работу')

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    #print(111)
    print('получил ответ от кнопки:', call.data)
    try:
        if call.data == "boss_ping":
            if call.message.date + 300 <= time.time():
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.id, text="Отряд не собран(")
            else:
                kb = telebot.types.InlineKeyboardMarkup(row_width=1)
                btn1 = telebot.types.InlineKeyboardButton(text="✅ я готов", callback_data='boss_ping')
                kb.add(btn1)
                # print(call.message.message_id, call.message.chat.id)
                edited_text = call.message.text.split('\n')
                this_usrname = call.from_user.username
                for i in range(len(edited_text)):
                    if edited_text[i].count(this_usrname) == 1:
                        edited_text[i] = '✅ ' + '@' + this_usrname
                        break
                edir = ''
                for i in range(len(edited_text)):
                    edir += edited_text[i] + '\n'
                if edir.count('✅') == 5:
                    edir += '\nОтряд собран!\n'
                    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=edir)
                    bot.reply_to(call.message, 'Запускайте боссса!')
                else:
                    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text=edir,
                                          reply_markup=kb)
        if call.data == 'settings_boss_ping':
            message = call.message
            ind = ind_check(call.from_user.id)
            if ind == -1:
                bot.reply_to(message,
                             f'@{call.from_user.username},для того чтобы настройки тыкать, нужно быть в моей базе')
            else:
                users[ind].boss_ping = not (users[ind].boss_ping)
                s = '✅Теперь я буду пинговать вас на боссов' if users[
                    ind].boss_ping else '⛔️Теперь я не буду пинговать вас на боссов'
                bot.answer_callback_query(call.id, text=s)
                update_data_users()
        if call.data == '+tuning':
            lab_tun(call.message)
        if call.data == '-tuning':
            lab_tun(call.message, -1)
        if call.data == '-quality':
            lab_qual(call.message, -1)
        if call.data == '+quality':
            lab_qual(call.message)
        if call.data == '+sharpening':
            lab_sharp(call.message)
        if call.data == '-sharpening':
            lab_sharp(call.message, -1)
    except AttributeError:
        print('', end='')




@bot.message_handler(commands=['update'])
def update(message):
    print(message.text)
    if message.from_user.id == 850966027:
        update_data_users()
        update_data_places()
        bot.reply_to(message, 'обновил')
    else:
        bot.reply_to(message, 'не для тебя команда')


@bot.message_handler(commands=['news'])
def news(message):

    print(message.text)
    ind = ind_check(message.from_user.id)
    if ind != -1:
        if users[ind].role == 'officer':
            _text = message.text.split('/news ')
            for i in users:
                bot.send_message(i.uid, _text[1] + '\n\nИ помните: 🏢 Империя заботится о вас!')


@bot.message_handler(commands=['set_timezone'])
def time_zone_reply(message):
    if message.chat.id != message.from_user.id:
        bot.reply_to(message, 'работает только в личке')
    else:
        ind = ind_check(message.from_user.id)
        if ind == -1:
            bot.reply_to(message,
                         'настройки только для зарегестрированных пользователей! Перешли мне свой профиль от @HyperionGameBot')
        else:
            sent = bot.reply_to(message,
                                'окей, отправь мне свою (зону времени?). Например чтобы у тебя показывало по МСК - отправь 3, так как время по МСК - UTC+3')
            bot.register_next_step_handler(sent, setting_time_zone)


# Handle '/start' and '/help'
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, """\
Добро пожаловать в 🏢 Эндимион!
я - бот, который поможет вам взаимодействовать с нашим городом.
отправь мне свой профиль с игры @HyperionGameBot.
""")


@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, """\
известная мне информация на данный момент:
/help - вызов меню с информацией
/prof_who - информация о профессии соотрядовцев
/me - информация о вас(желательно обновлять ее как можно чаще)
/settings - настройки (работает только в личных сообщениях бота)
/res - время до сбора ресурсов
/set_timezone - установить временную зону по UTC (работает только в личных сообщениях бота)
""")





@bot.message_handler(commands=['res'])
def res_command(message):
    res_time(message, False)


@bot.message_handler(commands=['prof_who'])
def res_command(message):
    s = ''
    for i in users:
        fl = True
        for j in cities:
            if i.city == j[1]:
                s += j[0]
                fl = False
        if fl:
            s += '❓'
        s += i.name + ': ' + i.prof.split("|Уровень: ")[0] + ', ' + i.prof.split("|Уровень: ")[
            1] + '\n' if i.prof != 'Неизвестно' else i.name + ': ' + i.prof + '\n'
    bot.reply_to(message, s)


@bot.message_handler(commands=['me'])
def send_welcome(message):
    send_profile(message)


@bot.message_handler(commands=['raids'])
def send_raids(message):
    send_raid(message, 2)


@bot.message_handler(content_types=['photo'])
def message_pocessing(message):
    message_processing(message, False)
    # bot.reply_to(message, 'абоба')


@bot.message_handler(commands=['settings'])
def settings_processing(message):
    ind = ind_check(message.from_user.id)
    if ind == -1:
        bot.reply_to(message, f'@{message.from_user.username},для того чтобы настройки тыкать, нужно быть в моей базе')
    else:
        kb = telebot.types.InlineKeyboardMarkup(row_width=1)
        s = '✅' if users[ind].boss_ping else '⛔️'
        btn1 = telebot.types.InlineKeyboardButton(text=f"Пинговать на боссов {s}", callback_data='settings_boss_ping')
        kb.add(btn1)
        bot.reply_to(message, 'настройки, которые вы можете изменить:', reply_markup=kb)


@bot.message_handler()
def messag_pocessing(message):
    # bot.reply_to(message, 'абоба2')
    message_processing(message, True)

def setting_time_zone(message):
    try:
        k = int(message.text)
        ind = ind_check(message.from_user.id)
        users[ind].timezone = k
        update_data_users()
        bot.reply_to(message, 'ага, записал')
    except:
        bot.reply_to(message, 'возникла ошибка, возможно ты ввел не число')


def message_processing(message, flag):
    try:
        print('сообщение: ', message.text[:10])
        _text = message.text if flag else message.caption
        # print(message.forward_from.id)
        if _text == 'Штыус, профиль':
            # print('huy52')
            send_profile(message)
        if message.from_user.id == 589732215 and int((random.randint(1, 1000))) == 7:
            bot.reply_to(message, 'АААААААААААААААА, ЖЕНЩИНА')
        if message.forward_from.id == 820567103:
            if _text.count('Если ты не хочешь слышать других игроков - нажми /toggle_radio') == 0:
                if _text.count('Группа отряда ') == 1:
                    if message.forward_date + 300 >= time.time():
                        s1 = 'Пинг!\n'
                        _text = re.split('Группа отряда | собралась.|Записавшиеся игроки:\n|\n', _text)[6:]
                        # print(_text)
                        # for i in range(len(users.txt)):
                        # print(users.txt[i].name)
                        kb = telebot.types.InlineKeyboardMarkup(row_width=1)
                        btn1 = telebot.types.InlineKeyboardButton(text="✅ я готов", callback_data='boss_ping')
                        kb.add(btn1)
                        for i in range(len(_text)):
                            fl = True
                            for j in range(len(users)):
                                if users[j].name in _text[i]:
                                    s1 += '@' + users[j].username + '\n'
                                    fl = False
                                    break
                            if fl:
                                s1 += _text[i] + '\n'
                        bot.reply_to(message, s1, reply_markup=kb)
                    else:
                        bot.reply_to(message, 'Вспомнил тоже, когда это было то?')
                elif 'Требуемые характеристики:' in _text and 'Качество: ' in _text:
                    if message.chat.id == message.from_user.id:
                        ind = ind_check(message.from_user.id)
                        if ind == -1:
                            bot.reply_to(message, 'сначала зарегестрируйся в боте')
                        else:
                            lab_func_st(message)
                elif _text.count('UID') == 1 and _text.count('Событие') == 1:
                    _text = re.split('👤 |, | \| |👨\u200d👨\u200d👧\u200d👦: |\n\n🗺:|UID: |\n', _text)
                    # print(_text)
                    if int(_text[-1]) != message.from_user.id:
                        bot.reply_to(message, 'скинь мне свой профиль, а не кого-то другого')
                    elif message.forward_date + 300 < time.time():
                        bot.reply_to(message, 'этому профилю больше 5 минут, я его не приму!')
                    else:
                        uid = int(_text[-1])
                        name = _text[3]
                        squad_name = _text[6]
                        # print(name, squad_name, uid)
                        hp, pp, mp, at, df = 0, 0, 0, 0, 0
                        for klol in _text:
                            if klol.count('🔮: ') == 1:
                                mp = int(re.split('🔮: |/', klol)[2])
                            if klol.count('❤️: ') == 1:
                                hp = int(re.split('❤️: |/', klol)[2])
                            if klol.count('💪: ') == 1:
                                pp = int(klol.split('💪: ')[1])
                            if klol.count('🛡: ') == 1:
                                df = int(klol.split('🛡: ')[1])
                            if klol.count('⚔️: ') == 1:
                                at = int(klol.split('⚔️: ')[1])
                        new = True
                        # print(_text)
                        for j in range(len(users)):
                            if users[j].uid == uid:
                                new = False
                                users[j].name = name
                                users[j].squad_name = squad_name
                                users[j].username = message.from_user.username
                                users[j].time = int(message.forward_date)
                                users[j].mana_p = mp
                                users[j].health_p = hp
                                users[j].power_p = pp
                                users[j].attack = at
                                users[j].deff = df
                                # print(_text[4][1:])
                                users[j].city = _text[4][2:]
                                bot.reply_to(message, 'Профиль обновлен!')
                                update_data_users()
                                break
                        if new:
                            users.append(User(message.from_user.id, message.from_user.username, name, squad_name,
                                              int(message.forward_date), 1, df, at, hp, pp, mp, 'user', True,
                                              _text[4][2:], 'Неизвестно', 1, 3))
                            print(str(users[-1]))
                            update_data_users()
                            bot.reply_to(message,
                                         'Добро пожаловать! Твой профиль добавлен в мою базу данных.\n/help - покажет доступные команды')

                elif _text.count('Здесь собираются игроки из отряда ') == 1:
                    if message.forward_date + 300 >= time.time():
                        _text = re.split(
                            'Здесь собираются игроки из отряда |, желающие победить |\n|Записавшиеся игроки:',
                            _text)
                        # print(_text)
                        s = [[]]
                        ind = 0
                        cnt = 0
                        _squad = _text[1]
                        for i in users:
                            fl = True
                            for j in _text:
                                if i.name in j:
                                    fl = False
                            if fl and _squad in i.squad_name:
                                if i.boss_ping:
                                    s[ind].append('@' + i.username)
                                else:
                                    s[ind].append(i.name)
                                cnt += 1
                                if cnt == 5:
                                    cnt = 0
                                    ind += 1
                                    s.append([])
                        for i in s:
                            strin = _squad + ', пишемся на босса ' + _text[2] + '\n\n'
                            if len(i)>0:
                                for k in i:
                                    strin += k + '\n'
                                bot.reply_to(message, strin)
                    else:
                        bot.reply_to(message, 'слишком старое сообщение')
                elif 'Собрано ' in _text:
                    res_time(message, True)
                elif 'Ты не записан на боссов' in _text:
                    bot.reply_to(message, 'Этот грех Аллах не простит')
                elif 'Сбросить: /reset_spec' in _text:
                    ind = ind_check(message.from_user.id)
                    if ind != -1:
                        fl = False
                        for i in users:
                            if i.prof_time == int(message.forward_date):
                                fl = True
                        if fl:
                            bot.reply_to(message, 'где-то я такое уже видел')
                        else:
                            if message.forward_date + 3600 >= int(time.time()):
                                _text = _text.split('\n')
                                s = ''
                                for i in _text:
                                    if 'Специализация: ' in i:
                                        s += i[16:] + '|'
                                    if 'Уровень: ' in i:
                                        s += i + ' '
                                    if 'Прогресс: ' in i:
                                        s += str(i.count('#'))
                                users[ind].prof = s
                                users[ind].prof_time = int(message.forward_date)
                                bot.reply_to(message, 'сохранил')
                                update_data_users()
                                users.sort(key=lambda User1: User1.prof)
                            else:
                                bot.reply_to(message, 'информация малость устарела, попробуй уложиться в 1 час')
                    else:
                        bot.reply_to(message, 'сначала отправь мне профиль')
                else:
                    for i in range(len(places)):
                        # print(places[i].name)
                        if places[i].name in _text and '↕️' in _text and '↔️' in _text and '🗺' in _text:
                            if message.forward_date + 21000 >= int(time.time()):
                                # print(str(places[i]))
                                if places[i].found:
                                    bot.reply_to(message, 'Уже нашли(')
                                else:
                                    places[i].found = True
                                    for l in range(len(zones)):
                                        if zones[l] in _text:
                                            places[i].zone = l
                                    zwyx = re.split('↕️: |  ↔️: |   🗺: |\n', _text)
                                    places[i].x = int(zwyx[3])
                                    places[i].y = int(zwyx[2])
                                    update_data_places()
                                    bot.reply_to(message, 'Записал')
                            else:
                                bot.reply_to(message, 'форвард малость староват, попробуй перезайти на точку')

    except AttributeError:
        print('', end='')


def ind_check(uid):
    ind = -1
    for j in range(len(users)):
        if uid == users[j].uid:
            ind = j
    return ind


def send_raid(message, type):
    s = ''
    for i in places:
        if i.found and i._type == type:
            s += f'{i.name}: ↕️:{i.y} ↔️:{i.x} {zones[i.zone]}'
    bot.reply_to(message, s)


def send_profile(message):
    s = 'Тебя нет в моей базе данных! Отправь профиль от @HyperionGameBot'
    ind = ind_check(message.from_user.id)
    if ind != -1:
        s2 = '✅Я пингую вас на боссов' if users[
            ind].boss_ping else '⛔️Я не пингую вас на боссов'
        s3 = ''
        fl = True
        for j in cities:
            if users[ind].city == j[1]:
                s3 = j[0]
                fl = False
        if fl:
            s3 = '❓'
        k = '\n'
        s = f'👤: {users[ind].name}, {s3} {users[ind].city}\n👨‍👨‍👧‍👦: {users[ind].squad_name}\nUID: `{users[ind].uid}`\n\n💪: {users[ind].power_p}, ❤️: {users[ind].health_p}, 🔮: {users[ind].mana_p}\n⚔️: {users[ind].attack}, 🛡: {users[ind].deff}\nПрофиль обновлен: {(int(time.time()) - users[ind].time + 1799) // 3600} часов назад\n\nПрофессия: {users[ind].prof.split("|")[0] + k + users[ind].prof.split("|")[1] if users[ind].prof != "Неизвестно" else users[ind].prof}\nОбновлена: {(int(time.time()) - users[ind].prof_time + 1799) // 3600} часов назад\n\n{s2}'
    bot.reply_to(message, s, parse_mode='Markdown')


def update_data_users():
    f = open('users.txt', 'w')
    for i in range(len(users)):
        f.write(str(users[i]))
    f.close()


def update_data_places():
    f = open('places.txt', 'w', encoding='utf-8')
    for i in range(len(places)):
        f.write(str(places[i]))
    f.close()


def res_time(message, fl):
    now_time = int(time.time())
    time_res = 8 * 60 * 60  # 8 часов в секундах
    ind = ind_check(message.from_user.id)
    # if ind != -1:
    if True:
        if fl:
            users[ind].res_time = message.forward_date
            update_data_users()
        if users[ind].res_time + time_res < now_time:
            bot.reply_to(message,
                         'По моим данным, ты уже можешь собрать ресурсы, либо пришли мне актуальный сбор ресурсов')
        else:
            _time = users[ind].res_time + time_res - now_time
            bot.reply_to(message,
                         f"Ты сможешь собрать ресурсы через {int(_time) // 3600} часов, {int(_time) % 3600 // 60} минут\nТочное время: {time.strftime('%H:%M:%S', time.localtime(users[ind].res_time + time_res + users[ind].timezone * 3600))}\n\n`⚒️ Собрать ресурсы`",
                         parse_mode='markdown')
    # else:
    #     bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAEMFodmPnuIXkaOMzpHeXsv-JOg7ChStwACpSUAAulB4EuYHdg8OtIqejUE',
    #                      protect_content=True)


def lab_func_st(message):
    _text = message.text.split('\n')
    # print(_text)
    koef = 1 if '✅ Надето' in message.text or '⛔️ Недостаточно характеристик' in message.text else 0
    _hp = int(_text[-7 + koef][:-2])
    s = f'{_text[0]}\n{_text[1]}\n{_text[3]}\n\n{_text[6]}\n'
    for i in _text:
        for j in shmot_dops:
            if j[0] in i:
                s += i
                if j[0] == j[-1]:
                    _stat = int(i.split(j[0])[1])
                    ed_stat = _hp / 200
                    low = int((_stat - 0.5) / ed_stat)
                    high = int((_stat + 0.5) / ed_stat)
                    if high > 25:
                        high = 25
                    if low == high:
                        s += f'   🔄: {high}\n'
                    else:
                        s += f'   🔄: {low}-{high}\n'
                else:
                    _stat = float(i.split(j[0])[1][:-1])
                    s += f'   🔄: {int(_stat / j[1])}\n'
    s += f'\n{_text[-8 + koef]}\n{_text[-7 + koef]}\n{_text[-5 + koef]}'
    bot.reply_to(message, s, reply_markup=lab_kb)


def lab_tun(message, mn=+1):
    # print(str(mn) + 'tun')
    _text = message.text.split('\n')
    s = f'{_text[0]}\n{_text[1]}\n'
    now_tun = int(_text[-1].split('Тюнинг: ')[1][:-1])
    next_tun = now_tun + mn * 5 if now_tun + mn * 5 >= -95 else -95
    s += f'+{round(int(_text[2][:-1] if not ("⚔️" in _text[2]) else _text[2][:-2]) / (100 + now_tun) * (100 + next_tun))}{_text[2][-1:] if not ("⚔️" in _text[2]) else _text[2][-2:]}\n\n{_text[4]}\n'
    for i in _text:
        for j in shmot_dops:
            if j[0] in i:
                if j[0] == j[-1]:
                    s += f'{j[0]} +{round(int(re.split(f"   🔄|{j[0]}", i)[1]) / (100 + now_tun) * (100 + next_tun))}   🔄: {i.split("   🔄: ")[1]}\n'
                else:
                    s += i + '\n'
    _hp = round(int(_text[-2][:-1]) / (100 + now_tun) * (100 + next_tun))
    s += f'\n{_text[-3]}\n{_hp} {_text[-2][-1]}\nТюнинг: {"+" if next_tun >= 0 else ""}{next_tun}%'
    bot.edit_message_text(chat_id=message.chat.id, message_id=message.id, text=s, reply_markup=lab_kb)
    # print(now_tun)


def lab_sharp(message, mn=+1):
    _text = message.text.split('\n')
    _s = re.split('\(|✨\)', _text[0])
    s = _s[0]
    #print(_s)
    sharp = 0 if len(_s) == 1 else int(_s[1])
    print(sharp)
    s += '\n' if sharp + mn <= 0 else f'(+{sharp + mn}✨)\n'
    s += _text[1] + '\n'
    if mn + sharp >= 0:
        s += f'+{round(int(_text[2][:-1] if not ("⚔️" in _text[2]) else _text[2][:-2]) * ((1.05) ** mn))}{_text[2][-1:] if not ("⚔️" in _text[2]) else _text[2][-2:]}\n'
    else:
        s += _text[2] + '\n'
    for i in range(3, len(_text)):
        s += _text[i] + '\n'

    bot.edit_message_text(chat_id=message.chat.id, message_id=message.id, text=s, reply_markup=lab_kb)


def lab_qual(message, mn=+1):
    _text = message.text.split('\n')


bot.infinity_polling()
polling()
