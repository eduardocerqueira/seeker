#date: 2023-01-24T17:06:05Z
#url: https://api.github.com/gists/41b886ee1c0e02b410184eef06205155
#owner: https://api.github.com/users/sstrixs

from pytube import YouTube
import requests
import webbrowser

a = 'https://github.com/sstrixs'

language = input('Choose language (English/Русский): ').lower()
source = input('Choose platform (instagram/youtube/vk): ').lower()

if language == 'русский' or 'рус' or 'heccrbq' or 'hec' and source == 'youtube' or 'ютуб' or 'нщгегиу' or '.ne,':
    link = str(input("Введите ссылку на видео: "))
    video = YouTube(link)
    quality = input('Выберите качество видео (Высокое/Низкое): ').lower()

    if quality == 'низкое':
        print('Видео загружаеться...')
        stream = video.streams.get_lowest_resolution()
        stream.download()
    if quality == 'высокое':
        print('Видео загружаеться...')
        stream = video.streams.get_highest_resolution()
        stream.download()
    print('Видео загружено!')
    print('Мой github https://github.com/sstrixs')
    webbrowser.open(a)

elif language == 'english' or 'eng' and source == 'youtube' or 'ютуб' or 'нщгегиу' or '.ne,':
    link = str(input("Type video link: "))
    video = YouTube(link)
    quality = input('Choose quality (High/Low): ').lower()

    if quality == 'low':
        print('Video is downloading...')
        stream = video.streams.get_lowest_resolution()
        stream.download()
    if quality == 'high':
        print('Video is downloading...')
        stream = video.streams.get_highest_resolution()
        stream.download()
    print('The video was successfully downloaded!')
    print('Visit my github https://github.com/sstrixs')
    webbrowser.open(a)

elif language == 'русский' or 'рус' or 'heccrbq' and source == 'instagram' or 'insta' or 'инстаграм' or 'инста':

    lg_link = input('Введите ссылку на пост: ')
    webbrowser.open(lg_link + 'media/?size=l')
    print('Ссылка для скачивания готова!')
    link2 = str(input('Введите ссылку из браузера: '))
    name = str(input('Введите имя для фотографии: '))
    print('Изображение загружается...')
    img = requests.get(link2)
    img_option = open(name + '.jpg', 'wb')
    img_option.write(img.content)
    img_option.close()
    print('Изображение успешно загружено!')
    print('Мой github https://github.com/sstrixs')
    webbrowser.open(a)

elif language == 'english' or 'eng' and source == 'instagram' or 'inst' or 'insta':

    lg_link = input('Type link: ')
    webbrowser.open(lg_link + 'media/?size=l')
    print('Download link ready!')
    link2 = str(input('Type link from browser: '))
    name = str(input('Type name you want: '))
    print('Image is downloading...')
    img = requests.get(link2)
    img_option = open(name + '.jpg', 'wb')
    img_option.write(img.content)
    img_option.close()
    print('Image was successfully downloaded!')
    print('Visit my github https://github.com/sstrixs')
    webbrowser.open(a)

elif language == 'english' or 'eng' and source == 'vk' or 'вк' or 'dr' or 'мл':

    link3 = str(input('Type link: '))
    name1 = str(input('Type name you want: '))
    print('Image is downloading...')
    img1 = requests.get(link3)
    img1_option = open(name1 + '.jpg', 'wb')
    img1_option.write(img1.content)
    img1_option.close()
    print('Image was successfully downloaded!')
    print('Visit my github https://github.com/sstrixs')
    webbrowser.open(a)

elif language == 'русский' or 'рус' or 'heccrbq' or 'hec' and source == 'vk' or 'вк' or 'dr' or 'мл':

    link4 = str(input('Введите ссылку на изображение: '))
    name5 = str(input('Введите имя для фотографии: '))
    print('Изображение загружается...')
    img3 = requests.get(link4)
    img3_option = open(name5 + '.jpg', 'wb')
    img3_option.write(img3.content)
    img3_option.close()
    print('Изображение успешно загружено!')
    print('Мой github https://github.com/sstrixs')
    webbrowser.open(a)
