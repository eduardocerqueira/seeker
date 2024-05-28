#date: 2024-05-28T16:48:51Z
#url: https://api.github.com/gists/745b86c61d4894c21e4a27956bbe51c6
#owner: https://api.github.com/users/hirdle

class BeeElephant:

    def __init__(self, bee, ele):
        self.bee_num = bee
        self.ele_num = ele

    def fly(self):
        if self.bee_num >= self.ele_num:
            return True
        else:
            return False

    def trumbet(self):
        if self.ele_num >= self.bee_num:
            return 'tu-tu-doo-doo!'
        else:
            return 'wzzzzz'

    def eat(self, meal, value):
        if meal == 'nectar':
            self.bee_num += value
            self.ele_num -= value
        else:
            self.bee_num -= value
            self.ele_num += value

        if self.bee_num > 100:
            self.bee_num = 100
        elif self.bee_num < 0:
            self.bee_num = 0

        if self.ele_num > 100:
            self.ele_num = 100
        elif self.ele_num < 0:
            self.ele_num = 0

    def get_parts(self):
        return (self.bee_num, self.ele_num)

be = BeeElephant(13, 87)
print(be.fly())
print(be.trumbet())
be.eat('nectar', 90)
print(be.trumbet())
print(be.get_parts())