#date: 2022-06-06T16:50:41Z
#url: https://api.github.com/gists/d01141143c1721fa2a4ce381e2ea9b88
#owner: https://api.github.com/users/FatihZor

from models import Person

person_data = {
    "name": "Ali",
    "surname": "Atabak",
    "age": 15,
    "gender": "Erkek"
}

Person(**person_data).save()


Person(
    name = "Ali",
    surname = "Atabak",
    gender = "Erkek",
    age = 15
).save()


person = Person()
person.name = "Ali"
person.surname = "Atabak"
person.gender = "Erkek"
person.age = 15
person.save()