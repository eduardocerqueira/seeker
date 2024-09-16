#date: 2024-09-16T17:08:45Z
#url: https://api.github.com/gists/f4b2d14e1c2552e58d2373ba74014c2b
#owner: https://api.github.com/users/zabelloalexandr

from pygments.formatters import other


class House:
    def __init__(self, name, number_of_floors):
        self.name = name
        self.number_of_floors = number_of_floors
        self.current_floor = 1

    def __eq__(self, other):
        return self.number_of_floors == other.new_floor

    def __str__(self):
        return f'{self.name} {self.number_of_floors}'

    def __lt__(self):
        return self.number_of_floors < other.new_floor
    def __len__(self):
        return self.number_of_floors




h1 = House('ЖК Эльбрус', 10)
h2 = House('ЖК Акация', 20)

# __str__
print(h1)
print(h2)

# __len__
print(len(h1))
print(len(h2))


