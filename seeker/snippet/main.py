#date: 2022-04-27T17:14:16Z
#url: https://api.github.com/gists/f87f554fe8c06c282e1a084574983d41
#owner: https://api.github.com/users/Igor-SeVeR

class CustomMeta(type):

    def __new__(cls, name, bases, dct):
        c_attrs = {}
        for key, value in dct.items():
            if not (key.startswith('__') and key.endswith('__')):
                c_attrs['custom_' + key] = value

            else:
                c_attrs[key] = value
        return super().__new__(cls, name, bases, c_attrs)

class CustomClass(metaclass=CustomMeta):
    x = 50

    def __init__(self, val=99):
        self.val = val

    def line(self):
        return 100

    def __str__(self):
        return "Custom_by_metaclass"

    def __setattr__(self, key, value):
        self.__dict__[f'custom_{key}'] = value


if __name__ == '__main__':
    inst = CustomClass()
    print(inst.custom_x)
    print(inst.custom_line())
    print(inst.custom_val)
    print(CustomClass.custom_x)
    print(str(inst) == "Custom_by_metaclass")
    inst.dynamic = "added later"
    print(inst.custom_dynamic == "added later")
