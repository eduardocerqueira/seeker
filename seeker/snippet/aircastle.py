#date: 2024-05-28T16:49:08Z
#url: https://api.github.com/gists/21ad374ce9597f1c389ddf04edf96556
#owner: https://api.github.com/users/hirdle

class AirCastle:
    def __init__(self, height, count, color):
        self.height = height
        self.count = count
        self.color = color

        self.func_comp = lambda x: (x.height, x.count, x.color)


    def change_height(self, value):
        self.height += value
        if self.height < 0:
            self.height = 0

    def __add__(self, other):
        self.count += other
        self.height += other // 5

    def __call__(self, opacity, *args, **kwargs):
        return self.height // opacity * self.count


    def __str__(self):
        return f'The AirCastle at an altitude of {self.height} meters is {self.color} with {self.count} clouds'


    def __gt__(self, other):
        return sorted([self, other], key=self.func_comp)[0] == other and other != self


    def __lt__(self, other):
        return sorted([self, other], key=self.func_comp)[0] == self and other != self


    def __ge__(self, other):
        min_val = sorted([self, other], key=self.func_comp)[0]
        return min_val == other or self == other


    def __le__(self, other):
        min_val = sorted([self, other], key=self.func_comp)[0]
        return min_val == self or self == other


    def __eq__(self, other):
        return (self.height == other.height and
                self.color == other.color and
                self.count == other.count)


    def __ne__(self, other):
        return not (self.height == other.height and
                    self.color == other.color and
                    self.count == other.count)




airc1 = AirCastle(5, 10, 'black')
airc3 = AirCastle(5, 10, 'black')
airc2 = AirCastle(5, 4, 'red')
print(airc3 > airc2)