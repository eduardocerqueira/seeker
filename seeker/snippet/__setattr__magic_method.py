#date: 2022-06-09T17:04:12Z
#url: https://api.github.com/gists/78eb5d9bbb26df7d74069bb933eccbf4
#owner: https://api.github.com/users/StephenFordham

class Employees(object):
    def __init__(self, name, age, location):
        self._name = name
        self._age = age
        self._location = location

    def __setattr__(self, key, value):
        if key in ['_name', '_location']:
            if not isinstance(value, str):
                raise TypeError('Only valid attributes of type string are accepted')
            else:
                self.__dict__[key] = value
        else:
            self.__dict__[key] = value


e1 = Employees('stephen', 30, 'Bournemouth')

for key, value in e1.__dict__.items():
    print('{}: {}'.format(key, value))
    
# Console Output
_name: stephen
_age: 30
_location: Bournemouth
