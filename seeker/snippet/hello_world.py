#date: 2022-08-25T15:04:51Z
#url: https://api.github.com/gists/1bd7e522cfdbcd74d499a43835922695
#owner: https://api.github.com/users/loujr

class HelloWorld:

    def __init__(self, name):
        self.name = name.capitalize()
       
    def sayHi(self):
        print "Hello " + self.name + "!"

hello = HelloWorld("world")
hello.sayHi()