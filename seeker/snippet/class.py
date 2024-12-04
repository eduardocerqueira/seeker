#date: 2024-12-04T16:55:56Z
#url: https://api.github.com/gists/5cb49fe01fc22d81f1c274e53bf6c116
#owner: https://api.github.com/users/TerskikhAndrey

class House:
    def __init__(self, name, number):
        self.name = str(name)
        self.number_of_floors = number

    def __len__(self):
        return self.number_of_floors

    def __str__(self):
        return  f'Название: {self.name}, количество этажей: {self.number_of_floors}'

    def __eq__(self, other):
        if isinstance(other.number_of_floors, int) and isinstance(other, House):
          return self.number_of_floors == other.number_of_floors
    def __lt__(self, other):
        if isinstance(other.number_of_floors, int) and isinstance(other, House):
            return self.number_of_floors < other.number_of_floors
    def __le__(self, other):
        if isinstance(other.number_of_floors, int):
            if isinstance(other, House):
                return self.number_of_floors <= other.number_of_floors
    def __gt__(self, other):
        if isinstance(other.number_of_floors, int):
            if isinstance(other, House):
                return self.number_of_floors > other.number_of_floors
    def __ge__(self, other):
        if isinstance(other.number_of_floors, int) and isinstance(other, House):
            return self.number_of_floors >= other.number_of_floors
    def __ne__(self, other):
        if isinstance(other.number_of_floors, int):
            if isinstance(other, House):
                return self.number_of_floors != other.number_of_floors
    def __add__(self, value):
        if isinstance(value, int):
            self.number_of_floors = self.number_of_floors + value
            return self
    def __radd__(self, value):
        return self.__add__(value)
    def __iadd__(self, value):
        if isinstance(value, int):
            self.number_of_floors += value
            return self




    def go_to(self, new_floor):
        floor = 0
        if new_floor < 1 or new_floor > self.number_of_floors:
            print('Такого экажа не существует')
        else:
            for floor in range(new_floor):
                print(floor + 1)
h1 = House('ЖК Горский', 10)



h2 = House('Домик в деревне', 20)


print(h1)
print(h2)
print(h1 == h2)
h1 = h1 + 10
print(h1)
print(h1 == h2)
h1 += 10
print(h1)
h2 = 10 + h2
print(h2)

print(h1 > h2)
print(h1 >= h2)
print(h1 < h2)
print(h1 <= h2)
print(h1 != h2)


