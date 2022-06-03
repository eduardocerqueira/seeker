#date: 2022-06-03T16:54:23Z
#url: https://api.github.com/gists/aa2e08f2c81031186d8b46c03372229f
#owner: https://api.github.com/users/sitkapo

class HelloWorld:

    def __init__(self, name):
        self.name = name.capitalize()
       
    def sayHi(self):
        print "Hello " + self.name + "!"

hello = HelloWorld("world")
hello.sayHi()