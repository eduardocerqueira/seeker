#date: 2025-08-28T17:01:22Z
#url: https://api.github.com/gists/1ba357953f9d94e6089ad9b9bc7a8821
#owner: https://api.github.com/users/kaliswag

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# --- ANSI Коды для цветов консоли (псевдо-темная тема) ---
RESET = "\033[0m"
BOLD = "\033[1m"
LIGHT_BLUE = "\033[94m"
LIGHT_GREEN = "\033[92m"
LIGHT_YELLOW = "\033[93m"
LIGHT_RED = "\033[91m"
DARK_GREY = "\033[90m"
WHITE_TEXT = "\033[97m"
BG_DARK = "\033[40m" # Черный фон

def enable_ansi_on_windows():
    """Включает поддержку ANSI escape-кодов в консоли Windows."""
    if sys.platform == "win32":
        os.system('') # Эта команда включает виртуальный терминал в Windows 10+

# ИСПРАВЛЕННАЯ ФУНКЦИЯ print_colored
def print_colored(text, color=WHITE_TEXT, bold=False, background=None, end='\n'): # Добавлен аргумент end='\n'
    """Печатает текст в указанном цвете."""
    style = BOLD if bold else ""
    bg_style = background if background else ""
    print(f"{bg_style}{color}{style}{text}{RESET}", end=end) # Передача end во встроенный print()

def get_yes_no_input(prompt):
    """Получает ответ пользователя 'да' или 'нет'."""
    while True:
        response = input(f"{LIGHT_YELLOW}{prompt}{RESET} (да/нет): ").lower().strip()
        if response in ['да', 'д']:
            return True
        elif response in ['нет', 'н']:
            return False
        else:
            print_colored("Пожалуйста, введите 'да' или 'нет'.", LIGHT_RED)

def get_user_choice(prompt, options):
    """Получает выбор пользователя из предложенных вариантов."""
    while True:
        print_colored(prompt, LIGHT_BLUE, bold=True)
        for i, option in enumerate(options, 1):
            print_colored(f"  {i}. {option}", WHITE_TEXT)
        try:
            choice = int(input(f"{LIGHT_YELLOW}Ваш выбор (номер): {RESET}").strip())
            if 1 <= choice <= len(options):
                return choice
            else:
                print_colored("Неверный номер. Пожалуйста, выберите из списка.", LIGHT_RED)
        except ValueError:
            print_colored("Неверный ввод. Пожалуйста, введите номер.", LIGHT_RED)

def select_directory_gui():
    """Открывает графическое окно для выбора директории."""
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно Tkinter
    print_colored("Открытие окна выбора директории...", DARK_GREY)
    directory = filedialog.askdirectory(
        title="Выберите директорию с файлами для переименования"
    )
    root.destroy() # Уничтожаем окно Tkinter после выбора
    return directory

def rename_files_script():
    """Основная функция для переименования файлов."""
    enable_ansi_on_windows() # Включаем ANSI для Windows
    os.system('cls' if sys.platform == 'win32' else 'clear') # Очищаем консоль для чистого старта

    print_colored("--- Утилита для массового переименования файлов ---", LIGHT_BLUE, bold=True)
    print_colored("Разработчик: sohurtt", DARK_GREY)
    print_colored("Используется 'темная тема' для консоли.", DARK_GREY)

    # 1. Выбор директории
    while True:
        choice = get_user_choice(
            "Как вы хотите выбрать директорию?",
            ["Ввести путь вручную", "Выбрать через графическое окно (как в WinRAR)"]
        )

        directory_path = ""
        if choice == 1:
            directory_path = input(f"{LIGHT_YELLOW}Введите путь к директории с файлами: {RESET}").strip()
        else:
            directory_path = select_directory_gui()

        if not directory_path:
            print_colored("Выбор директории отменен или не выбран путь. Попробуйте еще раз.", LIGHT_RED)
        elif not os.path.isdir(directory_path):
            print_colored(f"Ошибка: Директория '{directory_path}' не найдена или не является директорией. Попробуйте еще раз.", LIGHT_RED)
        else:
            break
    
    print_colored(f"Выбрана директория: {directory_path}", LIGHT_GREEN)

    # 2. Выбор операции
    operation_choice = get_user_choice(
        "Что вы хотите сделать?",
        ["Добавить что-то к названию", "Удалить часть названия"]
    )
    operation_type = "add" if operation_choice == 1 else "remove"

    string_to_modify = input(f"{LIGHT_YELLOW}Введите строку, которую хотите {'добавить' if operation_type == 'add' else 'удалить'}: {RESET}").strip()
    if not string_to_modify:
        print_colored("Ошибка: Вы не ввели строку для изменения. Операция отменена.", LIGHT_RED)
        return

    # Обработка пробелов
    add_space_before = False
    add_space_after = False
    remove_leading_space = False
    remove_trailing_space = False
    add_to_start = False

    if operation_type == "add":
        position_choice = get_user_choice(
            "Куда добавить строку?",
            ["В начало названия файла", "В конец названия файла (перед расширением)"]
        )
        add_to_start = (position_choice == 1)

        add_space_before = get_yes_no_input(f"Добавить пробел ПЕРЕД строкой '{string_to_modify}'?")
        add_space_after = get_yes_no_input(f"Добавить пробел ПОСЛЕ строкой '{string_to_modify}'?")
    else: # remove operation
        remove_leading_space = get_yes_no_input(f"Учитывать ли пробел ПЕРЕД строкой '{string_to_modify}' при удалении?")
        remove_trailing_space = get_yes_no_input(f"Учитывать ли пробел ПОСЛЕ строки '{string_to_modify}' при удалении?")


    # Собираем список файлов
    try:
        all_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    except OSError as e:
        print_colored(f"Ошибка при чтении директории: {e}", LIGHT_RED)
        return

    if not all_files:
        print_colored("В выбранной директории нет файлов.", LIGHT_YELLOW)
        return

    # Ограничение до 200 файлов, как просил пользователь
    files_to_process = all_files[:200]
    if len(all_files) > 200:
        print_colored(f"Внимание: Найдено более 200 файлов. Будут обработаны только первые {len(files_to_process)}.", LIGHT_YELLOW)

    print_colored(f"\nБудут обработаны {len(files_to_process)} файлов.", LIGHT_BLUE, bold=True)
    print_colored("Предварительный просмотр изменений (не более 5 примеров):", LIGHT_BLUE)

    preview_changes = []
    renamed_preview_count = 0

    # Генерируем предварительный просмотр
    for old_filename in files_to_process:
        base_name, extension = os.path.splitext(old_filename)
        new_base_name = base_name

        if operation_type == "add":
            prefix = " " if add_space_before else ""
            suffix = " " if add_space_after else ""
            modified_string = f"{prefix}{string_to_modify}{suffix}"

            if add_to_start:
                new_base_name = modified_string + base_name
            else:
                new_base_name = base_name + modified_string
        else: # operation_type == "remove"
            # Формируем точную строку для удаления с учетом пробелов
            target_to_remove = string_to_modify
            if remove_leading_space and remove_trailing_space:
                target_to_remove = " " + string_to_modify + " "
            elif remove_leading_space:
                target_to_remove = " " + string_to_modify
            elif remove_trailing_space:
                target_to_remove = string_to_modify + " "
            
            new_base_name = new_base_name.replace(target_to_remove, "")
            # Если пользователь выбрал учитывать пробелы, предполагаем, что он хочет удалить ТОЛЬКО
            # если пробелы присутствуют. Иначе, это слишком сложно для пользователя.
            # Поэтому, просто замена target_to_remove.

        new_filename = new_base_name + extension

        if old_filename != new_filename:
            if renamed_preview_count < 5:
                preview_changes.append(f"  '{old_filename}' -> '{new_filename}'")
            renamed_preview_count += 1
    
    if renamed_preview_count == 0:
        print_colored("  (Ни один файл не будет изменен, так как нет совпадений или нечего добавлять).", DARK_GREY)
        if not get_yes_no_input("Продолжить все равно? (Возможно, вы ожидаете, что что-то изменится)"):
            print_colored("Операция отменена.", LIGHT_RED)
            return
    else:
        for change in preview_changes:
            print_colored(change, WHITE_TEXT)
        if renamed_preview_count > 5:
            print_colored(f"  ... и еще {renamed_preview_count - 5} файлов.", DARK_GREY)

        if not get_yes_no_input(f"\nВы уверены, что хотите применить эти изменения к {renamed_preview_count} файлам?"):
            print_colored("Операция отменена.", LIGHT_RED)
            return

    # Запускаем переименование
    print_colored("\nНачинаем переименование...", LIGHT_GREEN, bold=True)
    processed_count = 0
    renamed_count = 0

    for old_filename in files_to_process:
        processed_count += 1
        base_name, extension = os.path.splitext(old_filename)
        new_base_name = base_name

        if operation_type == "add":
            prefix = " " if add_space_before else ""
            suffix = " " if add_space_after else ""
            modified_string = f"{prefix}{string_to_modify}{suffix}"

            if add_to_start:
                new_base_name = modified_string + base_name
            else:
                new_base_name = base_name + modified_string
        else: # operation_type == "remove"
            target_to_remove = string_to_modify
            if remove_leading_space and remove_trailing_space:
                target_to_remove = " " + string_to_modify + " "
            elif remove_leading_space:
                target_to_remove = " " + string_to_modify
            elif remove_trailing_space:
                target_to_remove = string_to_modify + " "
            
            new_base_name = new_base_name.replace(target_to_remove, "")

        new_filename = new_base_name + extension

        # Прогресс переименования
        # Используем end='\r' для перезаписи строки и flush() для немедленного вывода
        print_colored(f"({processed_count}/{len(files_to_process)}) Обработка: '{old_filename}'...", DARK_GREY, end='\r')
        sys.stdout.flush() 

        if old_filename != new_filename:
            old_full_path = os.path.join(directory_path, old_filename)
            new_full_path = os.path.join(directory_path, new_filename)
            try:
                os.rename(old_full_path, new_full_path)
                # После успешного переименования, печатаем на новой строке
                print_colored(f"Переименовано: '{old_filename}' -> '{new_filename}'", LIGHT_GREEN) 
                renamed_count += 1
            except FileExistsError:
                print_colored(f"Ошибка: Не удалось переименовать '{old_filename}' в '{new_filename}', файл с таким именем уже существует.", LIGHT_RED)
            except OSError as e:
                print_colored(f"Ошибка переименования '{old_filename}' в '{new_filename}': {e}", LIGHT_RED)
        else:
            # Если изменений нет, тоже печатаем на новой строке
            print_colored(f"Пропущено: '{old_filename}' (нет изменений)", DARK_GREY)

    print_colored(f"\nПереименование завершено. Успешно переименовано {renamed_count} файлов.", LIGHT_BLUE, bold=True)
    print_colored(f"Всего обработано файлов (из первых 200): {processed_count}", DARK_GREY)

if __name__ == "__main__":
    try:
        rename_files_script()
    except Exception as e:
        print_colored(f"\nНеожиданная ошибка: {e}", LIGHT_RED, bold=True)
    finally:
        # Важно сбросить стили в конце, чтобы не повлиять на консоль после завершения скрипта
        print(RESET) 
        print_colored("\n--- Важные замечания ---", LIGHT_BLUE, bold=True)
        print_colored("1.  Резервные копии: ВСЕГДА делайте резервные копии важных файлов перед тем, как запускать любые скрипты для массового переименования.", LIGHT_RED)
        print_colored("    Хотя скрипт имеет защиту от ошибок и предварительный просмотр, ошибки случаются, и данные могут быть утеряны.", LIGHT_RED)
        print_colored("2.  Графический выбор директории: Функция выбора директории через графическое окно (как в WinRAR) использует библиотеку Tkinter.", WHITE_TEXT)
        print_colored("    Если окно не появляется или возникают ошибки, убедитесь, что Tkinter правильно установлен и настроен в вашей системе Python.", WHITE_TEXT)
        print_colored("3.  Темная тема: 'Темная тема' реализована с помощью ANSI escape-кодов. Она работает на большинстве современных терминалов (Linux, macOS, Windows Terminal, PowerShell).", WHITE_TEXT)
        print_colored("    На старых версиях командной строки Windows (cmd.exe до Windows 10) эти цвета могут не отображаться или отображаться некорректно. Настройки фона консоли должны быть темными для лучшего восприятия.", WHITE_TEXT)
        print_colored("4.  Чувствительность к регистру: Python по умолчанию чувствителен к регистру.", WHITE_TEXT)
        print_colored("    Если вы хотите удалить 'Bass', а в названии файла есть 'bass', скрипт не найдет совпадения. Вводите строку точно так, как она есть в названиях.", WHITE_TEXT)
        print_colored("5.  Ограничение 200 файлов: Скрипт обрабатывает только первые 200 файлов, найденных в выбранной директории.", WHITE_TEXT)
        print_colored("    Если в директории больше файлов, остальные будут проигнорированы. Это сделано для безопасности и производительности, согласно вашему запросу.", WHITE_TEXT)
        print_colored("6.  Расширения файлов: Скрипт корректно отделяет расширение файла (например, '.mp3', '.wav', '.jpg') и изменяет только основную часть имени файла.", WHITE_TEXT)
        print_colored("7.  Обработка ошибок: В скрипт включена базовая обработка ошибок (например, если директория не найдена, файл с новым именем уже существует, или возникли проблемы при переименовании).", WHITE_TEXT)
        print(RESET) # Сброс всех стилей в конце (повторно, на всякий случай)
