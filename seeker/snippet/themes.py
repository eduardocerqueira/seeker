#date: 2024-05-28T16:48:26Z
#url: https://api.github.com/gists/29a2d5c87a1da32b3aebe756a856534b
#owner: https://api.github.com/users/hirdle

class Themes():
    def __init__(self, init_list):
        self.themes_list = init_list

    def add_theme(self, theme):
        self.themes_list.append(theme)

    def shift_one(self):
        self.themes_list = [self.themes_list[-1]]+self.themes_list[:-1]

    def get_themes(self):
        return self.themes_list

    def reverse_order(self):
        self.themes_list.reverse()

    def get_first(self):
        return self.themes_list[0]


# tl = Themes(['weather', 'rain'])
# tl.add_theme('warm')
# print(tl.get_themes())
# tl.shift_one()
# print(tl.get_first())

tl = Themes(['sun', 'feeding'])
tl.add_theme('cool')
tl.shift_one()
print(tl.get_first())
tl.reverse_order()
print(tl.get_themes())

