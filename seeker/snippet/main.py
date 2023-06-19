#date: 2023-06-19T17:07:02Z
#url: https://api.github.com/gists/00fcdf8b22f22300f59cabb022e3680e
#owner: https://api.github.com/users/mypy-play

class Foo():
    @classmethod
    def something(cls) -> None:
        print("foo")
        
class Bar():
    @classmethod
    def something(cls) -> None:
        print("bar")
        
class Fake():
    @classmethod
    def not_something(cls) -> None:
        print("fake")


# Works        
my_list: list[type[Foo | Bar]] = [Foo, Bar]

for item in my_list:
    item.something()
    
    
# Does not work
my_fake_list: list[type[Foo | Bar | Fake]] = [Foo, Bar, Fake]

for item2 in my_fake_list:
    item2.something()
