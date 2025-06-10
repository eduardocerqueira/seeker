#date: 2025-06-10T17:04:05Z
#url: https://api.github.com/gists/3ce543ddcff6de14557817c207db5e8d
#owner: https://api.github.com/users/Hixenov

import random
import string
import sys

try:
    import pyperclip
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip"])
    import pyperclip

MAX_LENGTH = 24

def get_int_input(prompt, min_value, max_value):
    try:
        value = int(input(prompt))
        if min_value <= value <= max_value:
            return value
        else:
            print(f"Ошибка: введите число от {min_value} до {max_value}")
    except ValueError:
        print("Ошибка: введите целое число.")
    return None

def get_character_set():
    print("\nВыберите типы символов, которые будут использоваться:")
    print("1 - Только цифры")
    print("2 - Цифры и спецсимволы")
    print("3 - Цифры, спецсимволы и буквы")
    choice = input("Ваш выбор (1/2/3): ")

    if choice == '1':
        return string.digits
    elif choice == '2':
        return string.digits + string.punctuation
    elif choice == '3':
        return string.digits + string.punctuation + string.ascii_letters
    else:
        print("Неверный выбор. Используются только цифры по умолчанию.")
        return string.digits

def generate_random_string(length, characters):
    return ''.join(random.choice(characters) for _ in range(length))

def main():
    print("=== Генератор случайных строк ===")
    count = get_int_input(f"\nВведите длину строки (от 1 до {MAX_LENGTH}): ", 1, MAX_LENGTH)
    if count is None:
        return

    amount = get_int_input("Сколько строк сгенерировать? (от 1 до 100): ", 1, 100)
    if amount is None:
        return

    charset = get_character_set()
    results = [generate_random_string(count, charset) for _ in range(amount)]

    print("\nСгенерированные строки:")
    for i, res in enumerate(results, 1):
        print(f"{i}: {res}")

    try:
        pyperclip.copy('\n'.join(results))
        print("\n✅ Все строки скопированы в буфер обмена.")
    except Exception as e:
        print(f"\n⚠ Не удалось скопировать в буфер: {e}")

if name == "main":
    main()