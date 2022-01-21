#date: 2022-01-21T16:52:43Z
#url: https://api.github.com/gists/1e2148a2ed8793e909b5c990a00e62f7
#owner: https://api.github.com/users/glagolboris

from mendeleev import elements
import mendeleev
import telebot
import random
# [Название, Обозначение, Масса, Порядковый номер, Валентность]
l_nums = ['₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈','₉']
symbol_list = list('#$%&\"()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r\x0b\x0c')
hello_list_bot = ['Вам тоже привет! 😄', 'Вам тоже привет! Хоть я и виртуальный, но у меня все еще есть чувства и мне очень приятно! 😇', 'О, привет! Я готов к работе. 👀', 'Здравствуй, рад тебя видеть! 💫']
hello_list = ['здравствуйте', 'привет', 'ave', 'ас-саляму алейкум', 'алоха', 'мабухай', 'ола', 'хола', 'доброе утро', 'добрый день', 'добрый вечер', 'доброй ночи', 'здравия желаю', 'приуэт', 'здрасьте', 'дратути', 'ку', 'здарова', 'хелло']
hau_list = ['как ты', 'как твои дела', 'дела']
name_list = ['имя', 'ник', 'зовут', 'звать', 'называть', 'называют']
name_list_bot = ['Меня зовут Борий, но вы можете называть меня, как хотите. 👨‍💻', 'Мой создатель назвал меня 000011100010111000011010000011100010111000101110000011100010111100011001000011100010111100101111, что означает Борий. Но вы можете называть меня как хотите. 😁', 'Меня зовут Борий, приятно познакомиться! 🙂']
creator_list = ['создал', 'создатель', 'сделал', 'автор', 'программист', 'отец', 'папа', 'папо', 'пап']
creator_list_bot = ['Меня создал Борис Глаголевский, чтобы я помогал тебе! 🧐', 'Мой создатель - Борис Глаголевский. Только тихо! 🤫', 'Так как я виртуальный у меня нет отца, но есть создатель. Этот создатель - Борис Глаголевский и я его считаю своим отцом. 🙃']

que_list = ['какая', 'какова', 'найди', 'можешь', 'плиз', 'пожалуйста', 'что', 'за', 'у', 'а', 'как', 'химия', 'химии', 'в', 'чему', 'равна']
bot = telebot.TeleBot(token='5093788766:AAH1irbWHQZF5JhO6pbSRfdaSQYqnf5Mjd0')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет, я твой виртуальный лаборант! 👨‍🔬 Чтобы узнать, что я могу - напиши "/help"!')

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, 'Я могу:\nУказать валентность элемента ✅,\nУказать обозначение элемента ✅,\nУказать молярную массу элемента ✅,\nУказать порядковый номер элемента ✅, \n✅ Помогу составить оксид любого элемента! ✅.\n📛Для этого просто напиши мне свой вопрос.📛')
@bot.message_handler(content_types=['text'])

def get_text_messages(message):
    for i in range(len(hello_list)):
        if hello_list[i] in message.text.lower():
            bot.send_message(message.chat.id, random.choice(hello_list_bot))
            break
    for i2 in range(len(hau_list)):
        if hau_list[i2] in message.text.lower():
            bot.send_message(message.chat.id, 'У меня все прекрасно, надеюсь у вас также! 🤗')
            break
    for i3 in range(len(name_list)):
        if name_list[i3] in message.text.lower() and 'меня' not in message.text.lower():
            bot.send_message(message.chat.id, random.choice(name_list_bot))
            break
    for i10 in range(len(creator_list)):
        if creator_list[i10] in message.text.lower():
            bot.send_message(message.chat.id, random.choice(creator_list_bot))
            break
    if 'я' in message.text.lower() and 'борис' in message.text.lower() or 'я' in message.text.lower() and 'боря' in message.text.lower():
        bot.send_message(message.chat.id, 'Папа?')
    elif 'вален' in message.text.lower():
        val_status = False
        mess_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                mess_list.remove(symbol_list[i4])
        true_mess = ''.join(mess_list)
        mess_list= true_mess.split()
        if len(mess_list) <= 1:
            pass
        else:
            for i5 in range(len(mess_list)):
                if 'валентность' in mess_list[i5]:
                    mess_list.pop(i5)
                    print(mess_list)
                    break
            for i8 in range(len(que_list)):
                if que_list[i8] in mess_list:
                    mess_list.remove(que_list[i8])
            while val_status == False:
                for i6 in range(len(mess_list)):
                    for i7 in range(len(elements)):
                        if mess_list[i6][0:-3].title() in elements[i7][0]:
                            text = 'Валентность элемента '+elements[i7][0]+' равна - '+elements[i7][4]+'. ✅'
                            bot.send_message(message.chat.id, text)
                            val_status = True
                            break
    elif 'знак' in message.text.lower() or 'знач' in message.text.lower():
        znak_status = False
        mess_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                mess_list.remove(symbol_list[i4])
        true_mess = ''.join(mess_list)
        mess_list= true_mess.split()
        if len(mess_list) <= 1:
            pass
        else:
            for i5 in range(len(mess_list)):
                if 'знак' in mess_list[i5] or 'знач' in mess_list[i5]:
                    mess_list.pop(i5)
                    
                    break
            for i8 in range(len(que_list)):
                if que_list[i8] in mess_list:
                    mess_list.remove(que_list[i8])
            print(mess_list)
            while znak_status == False:
                for i6 in range(len(mess_list)):
                    for i7 in range(len(elements)):
                        if mess_list[i6][0:-1].title() in elements[i7][0]:
                            text = 'Обозначение элемента '+elements[i7][0]+' - '+str(elements[i7][1])+'. ✅'
                            bot.send_message(message.chat.id, text)
                            znak_status = True
                            break 
    elif 'оксид' in message.text.lower() and message.text.lower != 'оксид кислорода':
        nom_status = False
        mess_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                mess_list.remove(symbol_list[i4])
        true_mess = ''.join(mess_list)
        mess_list = true_mess.split()
        for i5 in range(len(mess_list)):
            if 'оксид' in mess_list[i5]:
                mess_list.pop(i5)
                break
        for i8 in range(len(que_list)):
            if que_list[i8] in mess_list:
                mess_list.remove(que_list[i8])
        if len(mess_list) == 1:
            et = 0
            while nom_status == False:
                for i6 in range(len(mess_list)):
                    for i7 in range(len(elements)):
                        et += 1
                        if mess_list[i6][0:-3].title() in elements[i7][0]:
                            global epx, epx2
                            epx = i7
                            nom_status = True
                            break
            if len(mendeleev.elements_val[epx]) == 1:
                zn = mendeleev.lcm(mendeleev.elements_val[epx][0], 2)
                try:
                    x = zn // mendeleev.elements_val[epx][0]
                    y = zn // 2
                except Exception:
                    print(4)
                    bot.send_message(message.chat.id, 'К сожалению, нельзя составить оксид с этим элементом. 😞')
                else:
                    x = l_nums[x-1]
                    y = l_nums[y-1]
                    if x == '₁':
                        x = ''
                    if y == '₁':
                        y = ''
                    bot.send_message(message.chat.id, '' + mendeleev.elements_symb[epx] + '' + str(x) + 'O' + str(y) + ' ✅')

            else:
                print(3)
                bot.send_message(message.chat.id, 'Пожалуйста, введите запрос корректнее, мне не удалось вас понять 😥')
        elif len(mess_list) == 2:
            try:
                const = int(mess_list[1])
            except Exception:
                print(1)
                bot.send_message(message.chat.id, 'Пожалуйста, введите запрос корректнее, мне не удалось вас понять 😥')
            else:
                et = 0
                while nom_status == False:
                    for i6 in range(len(mess_list)):
                        for i7 in range(len(elements)):
                            et += 1
                            if mess_list[i6][0:-3].title() in elements[i7][0]:
                                global z2
                                global z
                                z = i7
                                if const in mendeleev.elements_val[z]:
                                    zn = mendeleev.lcm(const, 2)
                                    try:
                                        x = zn // const
                                        y = zn // 2
                                    except ZeroDivisionError:
                                        print(2)
                                        bot.send_message(message.chat.id, 'К сожалению, нельзя составить оксид с этим элементом. 😞')
                                    else:
                                        x = l_nums[x - 1]
                                        y = l_nums[y - 1]
                                        if x == '₁':
                                            x = ''
                                        if y == '₁':
                                            y = ''
                                        bot.send_message(message.chat.id, '' + mendeleev.elements_symb[z] + '' + str(x) + 'O' + str(y) + ' ✅')
                                        nom_status = True
                                        break
                                else:
                                    if nom_status != True:
                                        bot.send_message(message.chat.id, 'Пожалуйста, введите запрос корректнее, мне не удалось вас понять 😥')
                                        print(5)
                                    nom_status = True
                                    break



    elif 'номер' in message.text.lower() or 'поряд' in message.text.lower():
        nom_status = False
        mess_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                mess_list.remove(symbol_list[i4])
        true_mess = ''.join(mess_list)
        mess_list = true_mess.split()
        if len(mess_list) <= 1:
            pass
        else:
            for i5 in range(len(mess_list)):
                if 'номер' in mess_list[i5] or 'поряд' in mess_list[i5]:
                    mess_list.pop(i5)
                    
                    break
            for i8 in range(len(que_list)):
                if que_list[i8] in mess_list:
                    mess_list.remove(que_list[i8])
            print(mess_list)
            while nom_status  == False:
                for i6 in range(len(mess_list)):
                    for i7 in range(len(elements)):
                        if mess_list[i6][0:-3].title() in elements[i7][0]:
                            text = 'Атомный номер элемента '+elements[i7][0]+' - '+str(elements[i7][3])+'. ✅'
                            bot.send_message(message.chat.id, text)
                            nom_status = True
                            break 
    
    elif 'масса' in message.text.lower():
        mas_status = False
        mess_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                mess_list.remove(symbol_list[i4])
        true_mess = ''.join(mess_list)
        mess_list= true_mess.split()
        if len(mess_list) <= 1:
            pass
        else:
            for i5 in range(len(mess_list)):
                if 'масса' in mess_list[i5]:
                    mess_list.pop(i5)
                    print(mess_list)
                    break
            for i8 in range(len(que_list)):
                if que_list[i8] in mess_list:
                    mess_list.remove(que_list[i8])
            while mas_status == False:
                for i6 in range(len(mess_list)):
                    for i7 in range(len(elements)):
                        if mess_list[i6][0:-3].title() in elements[i7][0]:
                            text = 'Молярная масса '+elements[i7][0]+' равна - '+elements[i7][2]+'. ✅'
                            bot.send_message(message.chat.id, text)
                            mas_status = True
                            break
                if mas_status == True:
                    break
    elif 'табл' in message.text.lower() or 'систем' in message.text.lower():
        photo = open('5_uglerodnyĭ-shovinizm.jpg', 'rb')
        bot.send_photo(message.chat.id, photo)
    elif len(message.text.lower().split()) == 1:
        if message.text.title() in mendeleev.elements_names:
            el_name_indx = mendeleev.elements_names.index(message.text.title())
            bot.send_message(message.chat.id, 'Элемент - '+elements[el_name_indx][0]+' ('+elements[el_name_indx][1]+'). ✅\nЕго молярная масса - '+str(elements[el_name_indx][2])+' моль. ⚜️\nВалентность - '+elements[el_name_indx][4]+' ⚜️\nАтомный номер элемента - '+str(elements[el_name_indx][3])+'. ⚜️')
        elif message.text.title() in mendeleev.elements_symb:
            el_name_indx = mendeleev.elements_symb.index(message.text.title())
            bot.send_message(message.chat.id, 'Элемент - '+elements[el_name_indx][0]+' ('+elements[el_name_indx][1]+'). ✅\nЕго молярная масса - '+str(elements[el_name_indx][2])+' моль. ⚜️\nВалентность - '+elements[el_name_indx][4]+' ⚜️\nАтомный номер элемента - '+str(elements[el_name_indx][3])+'. ⚜️')
        
        elif message.text.title() in mendeleev.elements_numb:
            el_name_indx = mendeleev.elements_numb.index(message.text.title())
            bot.send_message(message.chat.id, 'Элемент - '+elements[el_name_indx][0]+' ('+elements[el_name_indx][1]+'). ✅\nЕго молярная масса - '+str(elements[el_name_indx][2])+' моль. ⚜️\nВалентность - '+elements[el_name_indx][4]+' ⚜️\nАтомный номер элемента - '+str(elements[el_name_indx][3])+'. ⚜️')
    elif 'номер' in message.text.lower():
        numb_status = False
        numb_list = list(message.text.lower())
        for i4 in range(len(symbol_list)):
            if symbol_list[i4] in mess_list:
                numb_list.remove(symbol_list[i4])
        true_numb = ''.join(numb_list)
        numb_list= true_numb.split()
        if len(numb_list) <= 1:
            pass
        else:
            for i5 in range(len(numb_list)):
                if 'номер' in numb_list[i5]:
                    numb_list.pop(i5)
                    print(numb_list)
                    break
            for i8 in range(len(que_list)):
                if que_list[i8] in numb_list:
                    numb_list.remove(que_list[i8])
            while numb_status == False:
                for i6 in range(len(numb_list)):
                    for i7 in range(len(elements)):
                        if numb_list[i6][0:-1].title() in elements[i7][0]:
                            text = 'Номер '+elements[i7][0]+' равна - '+elements[i7][4]+'. ✅'
                            bot.send_message(message.chat.id, text)
                            numb_status = True
                            break
                if numb_status == True:
                    break
    
    
                         


if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)