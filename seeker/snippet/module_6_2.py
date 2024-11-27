#date: 2024-11-27T17:09:11Z
#url: https://api.github.com/gists/b3eec24b9a4b70125473fea74f77fbd7
#owner: https://api.github.com/users/LiHogg

# Класс транспорта
class Vehicle:
    # Допустимые цвета - атрибут класса
    __COLOR_VARIANTS = ['blue', 'red', 'green', 'black', 'white']

    def __init__(self, owner, model, color, engine_power):
        self.owner = owner  # Владелец транспорта
        self.__model = model  # Модель транспорта
        self.__color = color  # Цвет транспорта
        self.__engine_power = engine_power  # Мощность двигателя

    # Метод для получения модели
    def get_model(self):
        return f"Модель: {self.__model}"

    # Метод для получения мощности двигателя
    def get_horsepower(self):
        return f"Мощность двигателя: {self.__engine_power}"

    # Метод для получения цвета транспорта
    def get_color(self):
        return f"Цвет: {self.__color}"

    # Метод для вывода информации о транспорте
    def print_info(self):
        print(self.get_model())
        print(self.get_horsepower())
        print(self.get_color())
        print(f"Владелец: {self.owner}")

    # Метод для изменения цвета
    def set_color(self, new_color):
        if new_color.lower() in map(str.lower, self.__COLOR_VARIANTS):
            self.__color = new_color
        else:
            print(f"Нельзя сменить цвет на {new_color}")


# Класс седан, наследуется от Vehicle
class Sedan(Vehicle):
    # Лимит пассажиров
    __PASSENGERS_LIMIT = 5

    # Конструктор наследуется от Vehicle, добавления дополнительных атрибутов не требуется
    pass


# Проверка программы
# Создаем объект класса Sedan
vehicle1 = Sedan('Fedos', 'Toyota Mark II', 'blue', 500)

# Изначальные свойства
vehicle1.print_info()

# Меняем свойства
vehicle1.set_color('Pink')  # Недопустимый цвет
vehicle1.set_color('BLACK')  # Допустимый цвет
vehicle1.owner = 'Vasyok'  # Меняем владельца

# Проверяем изменения
vehicle1.print_info()
