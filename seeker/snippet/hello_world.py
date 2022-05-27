#date: 2022-05-27T17:13:01Z
#url: https://api.github.com/gists/0fb0acc5475f799d9b6a6d4bd3365074
#owner: https://api.github.com/users/Diiego300years

class HelloWorld:

    def __init__(self, name):
        self.name = name.capitalize()
       
    def sayHi(self):
        print "Hello " + self.name + "!"

hello = HelloWorld("world")
hello.sayHi()