#date: 2023-08-03T16:49:20Z
#url: https://api.github.com/gists/e0b8093b54529818518d8a4776978447
#owner: https://api.github.com/users/seefalert

import os
import pathlib
import time
from sys import platform
from time import sleep
import psutil
import subprocess

# Определяем папку с текущим скриптом
DIR = pathlib.Path(__file__).parent.resolve()

# Проверяем наличие папки "tests" и файла "my_code.py"
tests = os.path.join(DIR, 'tests')
try:
    n_tests = len(os.listdir(tests)) // 2
    open('my_code.py', encoding='utf-8')
except FileNotFoundError as error:
    # Выводим сообщение об ошибке, если папка или файл не найдены
    print('-' * 69, '\nОШИБКА 404')
    print('Папка с тестами должна называться - tests, а файл с кодом - my_code.py', '-' * 69, sep='\n')
    raise

# Определяем версию Python в зависимости от операционной системы
if platform == "linux" or platform == "linux2":
    python_version = 'python3'
elif platform == "darwin":
    python_version = 'python3'
elif platform == "win32":
    python_version = 'py'

# Запускаем тесты для каждого тестового файла
for i in range(1, n_tests + 1):
    process = psutil.Process(os.getpid())
    start_time = time.time()
    with open(os.path.join(tests, str(i)), encoding='utf-8') as test_file, open(os.path.join(tests, f'{str(i)}.clue'),
                                                                           encoding='utf-8') as clue_file:
        test_data = test_file.read().splitlines()
        result_bytes = subprocess.run([python_version, "my_code.py"], input='\n'.join(test_data).encode('utf-8'),
                                      capture_output=True).stdout
        result = result_bytes.decode('utf-8').strip().splitlines()
        correct = clue_file.read().strip().splitlines()

        if result != correct:
            # Выводим данные, если результаты не совпадают
            print(f"Test#{i} Input:")
            print('\n'.join(test_data))
            print(f"Test#{i} Expected Output:")
            print('\n'.join(correct))
            print(f"Test#{i} Actual Output:")
            print('\n'.join(result))

        # Проверяем результаты и выводим статистику о тесте
        assert result == correct, f"Test#{i}\n{'-' * 69}\nexpect:{repr(correct)}\nresult:{repr(result)}\n"
    end_time = time.time()
    elapsed_time = end_time - start_time
    memory_usage = process.memory_info().rss / 1024 / 1024
    print(
        f'Тест №{i} пройден(✓), время выполнения: {elapsed_time:.2f} секунд, использовано памяти: {memory_usage:.2f} MB')
    sleep(0.5)
