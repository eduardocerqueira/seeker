#date: 2022-12-12T16:48:01Z
#url: https://api.github.com/gists/fc2c257c98e201079346cf0ca3eb3404
#owner: https://api.github.com/users/karchh

# Домашка №5. Сдайте задание до: 12 дек., 20:00 +03:00 UTC
# Задача 1.
# Ускоренная обработка данных: lambda, filter, map, zip, enumerate, list comprehension
# Создайте программу для игры с конфетами человек против компьютера.
# Условие задачи: На столе лежит 150 конфет. Играют игрок против компьютера.
# Первый ход определяется жеребьёвкой. За один ход можно забрать не более чем 28 конфет.
# Все конфеты оппонента достаются сделавшему последний ход. Подумайте как наделить бота ""интеллектом""


# from random import *
#
# preview_text = ('На столе лежит 150 конфет. Играют игрок против компьютера.'
#                 'Первый ход определяется жеребьёвкой.\nЗа один ход можно забрать '
#                 'не более чем 28 конфет. Все конфеты оппонента достаются сделавшему последний ход.')
# print(preview_text)
#
# rnd_message = ['Бери', 'Не тупи, бери конфеты', 'Хватай из кучки', 'Да бери уже эти сраные конфеты',
#            'Бери не очкуй:)']
#
# def player_vs_player():
#     candies_total = 150
#     max_take = 28
#     count = 0
#     player_1 = input('\nКак тебя зовут?: ')
#     player_2 = input('\nНапиши имя соперника: ')
#
#     print(f'\nГоспода {player_1} и {player_2}, сейчас комп рандомно определит, кто будет первым ходить!\n')
#
#     x = randint(1, 2)
#     if x == 1:
#         lucky = player_1
#         loser = player_2
#     else:
#         lucky = player_2
#         loser = player_1
#     print(f'Игрок, {lucky} - ты ходишь первым!')
#
#     while candies_total > 0:
#         if count == 0:
#             step = int(input(f'\n{choice(rnd_message)} {lucky} = '))
#             if step > candies_total or step > max_take:
#                 step = int(input(
#                     f'\nБери не больше {max_take} конфет {lucky}, трай эгэн: '))
#             candies_total = candies_total - step
#         if candies_total > 0:
#             print(f'\nв кучке еще {candies_total}')
#             count = 1
#         else:
#             print('Все, конфеты закончились!')
#
#         if count == 1:
#             step = int(input(f'\n{choice(rnd_message)}, {loser} '))
#             if step > candies_total or step > max_take:
#                 step = int(input(
#                     f'\nБери не больше {max_take} конфет {loser}, трай эгэн: '))
#             candies_total = candies_total - step
#         if candies_total > 0:
#             print(f'\nв кучке еще {candies_total}')
#             count = 0
#         else:
#             print('Все, конфеты закончились!')
#
#     if count == 1:
#         print(f'Игрок {loser} ты - ПОБЕДИЛ!')
#         print(f'Игрок {lucky} ты все ПРОЕ**Л!')
#     if count == 0:
#         print(f'Игрок {lucky} ты - ПОБЕДИЛ')
#         print(f'Игрок {loser} ты все ПРОЕ**Л!')
# player_vs_player()
#




# Задача 3. Реализуйте RLE алгоритм: реализуйте модуль сжатия и восстановления данных.
# Входные и выходные данные хранятся в отдельных текстовых файлах.


with open('text.txt', 'w', encoding='UTF-8') as file:
    file.write(input('Напишите текст необходимый для сжатия: '))
with open('text.txt', 'r') as file:
    my_text = file.readline()
    text_compression = my_text.split()

print(my_text)

def rle_encode(text):
    enconding = ''
    prev_char = ''
    count = 1
    if not text:
        return ''
    for char in text:
        if char != prev_char:
            if prev_char:
                enconding += str(count) + prev_char
            count = 1
            prev_char = char
        else:
            count += 1
    else:
        enconding += str(count) + prev_char
        return enconding

text_compression = rle_encode(my_text)

with open('text_compression.txt', 'w', encoding='UTF-8') as file:
    file.write(f'{text_compression}')
print(text_compression)

