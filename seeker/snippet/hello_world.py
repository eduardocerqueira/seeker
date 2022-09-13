#date: 2022-09-13T17:17:12Z
#url: https://api.github.com/gists/5985857989d3b2cbc0236ff82492655b
#owner: https://api.github.com/users/pkrawczyk437

class HelloWorld:

    def __init__(self, name):
        self.name = name.capitalize()
       
    def sayHi(self):
        print "Hello " + self.name + "!"

hello = HelloWorld("world")
hello.sayHi()