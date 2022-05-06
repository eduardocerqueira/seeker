#date: 2022-05-06T17:10:28Z
#url: https://api.github.com/gists/5d58bba45f3e3737495404f76e1973f4
#owner: https://api.github.com/users/justxor

class Test:
    def __getattr__(self, item):
        print(f'__getattr__({item})')
        return -1

    def __getattribute__(self, item):
        print(f'__getattribute__({item})')
        if item == 'y':  # запретим получать y
            raise AttributeError
        return super().__getattribute__(item)

# зададим x и y
t = Test()
t.x = 10
t.y = 20

print(t.x)  # __getattribute__(x) 10
print(t.y)  # __getattribute__(y) __getattr__(y) -1
print(t.z)  # __getattribute__(z) __getattr__(z) -1

