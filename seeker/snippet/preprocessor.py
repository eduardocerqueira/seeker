#date: 2021-10-08T17:16:24Z
#url: https://api.github.com/gists/293b49a8c53e6a766c397bfec24c6792
#owner: https://api.github.com/users/rahmaevao

"""
# Препроцессор для скриптов Galileosky

Скрипты в GalileoSky ужасны, так как нельзя их делать из нескольких
файлов-модулей.

Для этого нужны данные скрипты, которые ищут файл main и в нем ищет все
#include директивы. По данным директивам
скрипт ищет модули и просто включает их в основной файл.
При этом защищая от повторных включений.

Пробег по файлу идет до тех пор, пока за пробег не будет найден ни
один #include.

Итоговый файл копируется в буфер обмена.
"""
import shutil
import os
import codecs
import pyperclip
import argparse


if __name__ == '__main__':

    main_filename = 'main.c'
    tempfilenames = []  # Для удаления в конце
    include_files = set()  # Для защиты от повторных включений

    while (True):
        main_file = codecs.open(main_filename, 'r', 'utf-8')
        ret_filename = 'ret' + main_filename
        ret_file = codecs.open(ret_filename, 'w', 'utf-8')
        tempfilenames.append(ret_filename)
        for main_line in main_file:
            if (main_line.find('#include') == -1):
                ret_file.write(main_line)
            else:
                include_filename = main_line.split('"')[1]
                if include_filename in include_files:
                    continue
                else:
                    include_files.add(include_filename)
                    include_file = codecs.open(include_filename, 'r', 'utf-8')
                    for include_line in include_file:
                        ret_file.write(include_line)
                    include_file.close()

        main_file.close()
        ret_file.close()

        # Поиск в рет файле директив инклюд для выхода из цикла
        find_include_flag = False
        ret_file = codecs.open(ret_filename, 'r', 'utf-8')
        for ret_line in ret_file:
            if (ret_line.find('#include') != -1):
                find_include_flag = True
                break
        if not find_include_flag:
            ret_file.close()
            break
        ret_file.close()
        main_filename = ret_filename


    # Копирование в буфер обмена
    global_string = ""
    main_ret_file = codecs.open(ret_filename, 'r', 'utf-8')
    for line in main_ret_file:
        global_string += line
    pyperclip.copy(global_string)
    main_ret_file.close()

    
    # Вычистка временный файлов
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--clear', action='store_true', default=False)
    if (not parser.parse_args().clear):
        shutil.copyfile(ret_filename, 'ret_file.c')

    for temp_file in tempfilenames:
        path = os.path.join(os.getcwd(), temp_file)
        os.remove(path)
    
    print("💚 The assembly was successful. Your code is on the clipboard. Paste it in the configurator.")
