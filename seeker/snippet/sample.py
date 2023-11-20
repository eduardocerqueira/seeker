#date: 2023-11-20T16:27:56Z
#url: https://api.github.com/gists/c936529527016423b86f5e952923f5a1
#owner: https://api.github.com/users/parttimenerd

"""
Sample file to test the trace module.

This should print:

    ...
    ********** Trace Results **********
    Used classes:
      only static init:
      not only static init:
       __main__.TestClass
         <static init>
         <static>static_method
         __init__
         class_method
         instance_method
    Free functions:
      all_methods
      free_function
      log
      teardown

License: MIT
"""

import trace

trace.setup(r".*")


def log(message: str):
    print(message)


class TestClass:
    x = 100

    def __init__(self):
        log("instance initializer")

    def instance_method(self):
        log("instance method")

    @staticmethod
    def static_method():
        log("static method")

    @classmethod
    def class_method(cls):
        log("class method")


def free_function():
    log("free function")


def all_methods():
    log("all methods")
    TestClass().instance_method()
    TestClass.static_method()
    TestClass.class_method()
    free_function()


all_methods()
