#date: 2022-12-21T16:35:05Z
#url: https://api.github.com/gists/34d95dff97282b4c228072004b9f7b42
#owner: https://api.github.com/users/sukubhattu

import json

class JsonMixin(object):
    def to_json(self):
        dict_items = self.__dict__
        return json.dumps(dict_items)


class Person():
    def __init__(self, name, salutation):
        self.name = name
        self.salutation = salutation
    
    def get_full_name(self):
        return f'{self.salutation} {self.name}'
        

class Employee(JsonMixin, Person):
    def __init__(self, name, salutation, skills, hobby):
        super().__init__(name, salutation)
        self.skills = skills
        self.hobby = hobby
    

e = Employee('John', 'Mr', ['Python', 'Java'], 'Cricket')
print(e.to_json())
