#date: 2022-05-06T17:09:10Z
#url: https://api.github.com/gists/059fad2bc978f0c70f4220f780f173fd
#owner: https://api.github.com/users/justxor

class Test:
def __getattr__(self, item):
print(f'__getattr__({item})')
return -1
t = Test()
# зададим x и y
t.x = 10
setattr(t, 'y', 33)
print(t.x) # 10
print(t.y) # 33
print(t.z) # __getattr__(z) -1