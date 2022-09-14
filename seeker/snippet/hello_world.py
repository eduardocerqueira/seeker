#date: 2022-09-14T17:24:54Z
#url: https://api.github.com/gists/739690d14175c43debcd72962eef7a45
#owner: https://api.github.com/users/pkrawczyk437

class HelloWorld:

    def __init__(self, name):
        self.name = name.capitalize()
       
    def sayHi(self):
        print "Hello " + self.name + "!"

hello = HelloWorld("world")
hello.sayHi()