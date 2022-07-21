#date: 2022-07-21T16:50:46Z
#url: https://api.github.com/gists/d84021f81f6ab06b6f882374c37a1df9
#owner: https://api.github.com/users/dublado

name = "Eric"
age = 74
f"Hello, {name}. You are {age}."


f"{2 * 37}"

def to_lowercase(input):
    return input.lower()

name = "Eric Idle"
f"{to_lowercase(name)} is funny."

class Comedian:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

    def __str__(self):
        return f"{self.first_name} {self.last_name} is {self.age}."

    def __repr__(self):
        return f"{self.first_name} {self.last_name} is {self.age}. Surprise!"
      
new_comedian = Comedian("Eric", "Idle", "74")
f"{new_comedian}"


f"{new_comedian}"

f"{new_comedian!r}"



>>> name = "Eric"
>>> profession = "comedian"
>>> affiliation = "Monty Python"
>>> message = (
...     f"Hi {name}. "
...     f"You are a {profession}. "
...     f"You were in {affiliation}."
... )
>>> message
'Hi Eric. You are a comedian. You were in Monty Python.'

message = f"""
    Hi {name}. 
    You are a {profession}. 
    You were in {affiliation}.
"""


>>> f"The \"comedian\" is {name}, aged {age}."
'The "comedian" is Eric Idle, aged 74.'


>>> comedian = {'name': 'Eric Idle', 'age': 74}
>>> f"The comedian is {comedian['name']}, aged {comedian['age']}."
The comedian is Eric Idle, aged 74.


>>> f"{{70 + 4}}"
'{70 + 4}'



https://realpython.com/python-f-strings/
  https://realpython.com/python-string-formatting/
    https://realpython.com/python-web-scraping-practical-introduction/