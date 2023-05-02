#date: 2023-05-02T17:02:27Z
#url: https://api.github.com/gists/993336fa60503ebe559a20163b91673e
#owner: https://api.github.com/users/megaamfibia

import pytest
from selenium.webdriver.chrome import webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.fixture(autouse=True)
def test_first():
    pytest.driver = webdriver.Chrome('D:/distrib/chrome/chromedriver.exe')
    # Переходим на страницу авторизации
    pytest.driver.get('http://petfriends.skillfactory.ru/login')
    pytest.driver.find_element(By.NAME, 'email').send_keys('azarova.a@yandex.ru')
    pytest.driver.find_element(By.NAME, 'pass').send_keys('12345678')
    pytest.driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()
    assert pytest.driver.find_element(By.TAG_NAME, 'h1').text == "PetFriends"
    pytest.driver.find_element(By.XPATH, "//a[@href='/my_pets']").click()
    yield
    pytest.driver.quit()


def test_all_pets_are_presents():
    pytest.driver.find_element(By.XPATH, "//a[@href='/my_pets']").click()
    """Проверяем, что на странице со списком моих питомцев присутствуют все питомцы"""

    WebDriverWait(pytest.driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".\\.col-sm-4.left")))

    # Сохраняем в переменную stat элементы статистики
    stat = pytest.driver.find_elements(By.CSS_SELECTOR, ".\\.col-sm-4.left")

    WebDriverWait(pytest.driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".table.table-hover tbody tr")))

    # Сохраняем в переменную pets элементы карточек питомцев
    pets = pytest.driver.find_elements(By.CSS_SELECTOR, '.table.table-hover tbody tr')

    # Получаем количество питомцев из данных статистики
    number = stat[0].text.split('\n')
    number = number[1].split(' ')
    number = int(number[1])

    # Получаем количество карточек питомцев
    number_of_pets = len(pets)

    # Проверяем что количество питомцев из статистики совпадает с количеством карточек питомцев
    assert number == number_of_pets


def test_half_pets_photo_available():
    """ Проверяем, что на странице со списком моих питомцев хотя бы у половины питомцев есть фото """

    WebDriverWait(pytest.driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".\\.col-sm-4.left")))

    # Сохраняем в переменную statistic элементы статистики
    stat = pytest.driver.find_elements(By.CSS_SELECTOR, ".\\.col-sm-4.left")

    # Сохраняем в переменную images элементы с атрибутом img
    images = pytest.driver.find_elements(By.CSS_SELECTOR, '.table.table-hover img')

    # Получаем количество питомцев из данных статистики
    number = stat[0].text.split('\n')
    number = number[1].split(' ')
    number = int(number[1])

    # Находим половину от количества питомцев
    half = number // 2

    # Находим количество питомцев с фотографией
    number_of_photos = 0
    for i in range(len(images)):
        if images[i].get_attribute('src') != '':
            number_of_photos += 1

    # Проверяем что количество питомцев с фотографией больше или равно половине количества питомцев
    assert number_of_photos >= half
    print(f'количество фото: {number_of_photos}')
    print(f'Половина от числа питомцев: {half}')


def test_no_duplicate():
    """Проверяем, что на странице со списком моих питомцев нет повторяющихся питомцев"""

    # Устанавливаем явное ожидание
    WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located
                                           ((By.CSS_SELECTOR, ".table.table-hover tbody tr")))

    # Сохраняем в переменную pet_data элементы с данными о питомцах
    pet_data = pytest.driver.find_elements(By.CSS_SELECTOR, '.table.table-hover tbody tr')

    # Перебираем данные из pet_data, оставляем имя, возраст, и породу остальное меняем на пустую строку
    # и разделяем по пробелу.
    list_data = []
    for i in range(len(pet_data)):
        data_pet = pet_data[i].text.replace('\n', '').replace('×', '')
        split_data_pet = data_pet.split(' ')
        list_data.append(split_data_pet)

    # Склеиваем имя, возраст и породу, получившиеся склеенные слова добавляем в строку
    # и между ними вставляем пробел
    line = ''
    for i in list_data:
        line += ''.join(i)
        line += ' '

    # Получаем список из строки line
    list_line = line.split(' ')

    # Превращаем список в множество
    set_list_line = set(list_line)

    # Находим количество элементов списка и множества
    a = len(list_line)
    b = len(set_list_line)

    # Из количества элементов списка вычитаем количество элементов множества
    result = a - b

    # Если количество элементов == 0 значит карточки с одинаковыми данными отсутствуют
    assert result == 0


def test_all_pets_have_diff_names():
    """Проверяем, что на странице со списком моих питомцев, у всех питомцев разные имена"""

    WebDriverWait(pytest.driver, 10).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".table.table-hover tbody tr")))
    # Сохраняем в переменную pet_data элементы с данными о питомцах
    pet_data = pytest.driver.find_elements(By.CSS_SELECTOR, '.table.table-hover tbody tr')

    # Перебираем данные из pet_data, оставляем имя, возраст, и породу остальное меняем на пустую строку
    # и разделяем по пробелу. Выбираем имена и добавляем их в список pets_name.
    pets_name = []
    for i in range(len(pet_data)):
        data_pet = pet_data[i].text.replace('\n', '').replace('×', '')
        split_data_pet = data_pet.split(' ')
        pets_name.append(split_data_pet[0])

    # Перебираем имена и если имя повторяется то прибавляем к счетчику r единицу.
    # Проверяем, если r == 0 то повторяющихся имен нет.
    r = 0
    for i in range(len(pets_name)):
        if pets_name.count(pets_name[i]) > 1:
            r += 1
    assert r == 0
    print(r)
    print(pets_name)


def test_presence_name_age_type():
    """Проверяем, что на странице со списком моих питомцев, у всех питомцев есть имя, возраст и порода"""

    WebDriverWait(pytest.driver, 10).until(EC.visibility_of_element_located
                                           ((By.CSS_SELECTOR, ".table.table-hover tbody tr")))
    # Сохраняем в переменную pet_data элементы с данными о питомцах
    pet_data = pytest.driver.find_elements(By.CSS_SELECTOR, '.table.table-hover tbody tr')

    # Перебираем данные из pet_data, оставляем имя, возраст, и породу остальное меняем на пустую строку
    # и разделяем по пробелу. Находим количество элементов в получившемся списке и сравниваем их
    # с ожидаемым результатом
    for i in range(len(pet_data)):
        data_pet = pet_data[i].text.replace('\n', '').replace('×', '')
        split_data_pet = data_pet.split(' ')
        result = len(split_data_pet)
        assert result == 3
