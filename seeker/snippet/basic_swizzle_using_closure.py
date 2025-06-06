#date: 2025-06-06T16:48:52Z
#url: https://api.github.com/gists/cc48b01ee289cd914f48b5405d2293bd
#owner: https://api.github.com/users/RampantLions

class MyClass:
    def say(self):
        return "Original"

original_method = MyClass.say

def swizzled_say(self):
    result = original_method(self)
    return f"{result} + Swizzled"

MyClass.say = swizzled_say

print(MyClass().say())  # "Original + Swizzled"
