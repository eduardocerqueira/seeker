#date: 2021-09-20T17:12:28Z
#url: https://api.github.com/gists/4e328e4a70c7e64aafa12b9e37fd2f59
#owner: https://api.github.com/users/SahbiOuali13

import re
import string
from pathlib import Path
from typing import List

from tinydb import TinyDB, Query, where


class User:

    DB = TinyDB(Path(__file__).resolve().parent / "db.json", indent=4)

    def __init__(
        self, first_name: str, last_name: str, phone_number: str = "", address: str = ""
    ) -> None:
        self.first_name = first_name
        self.last_name = last_name
        self.phone_number = phone_number
        self.address = address

    def __str__(self) -> str:
        return f"{self.full_name}\n{self.phone_number}\n{self.address}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.first_name}, {self.last_name}, {self.phone_number}, {self.address})"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def db_instance(self):
        return User.DB.get((where('first_name') == self.first_name) & (where('last_name') == self.last_name))


    def _check_phone_number(self) -> None:
        phone_number = re.sub(r"[+()\s]*", "", self.phone_number)
        # print(phone_digits)
        if len(phone_number) < 10 or not phone_number.isdigit():
            raise ValueError(f"Invalid phone number: {phone_number}. ")

    def _check_names(self) -> None:
        if not (self.first_name and self.last_name):
            raise ValueError("Last name and first name can't be empty")

        special_characters = string.punctuation + string.digits

        if set(self.first_name + self.last_name).intersection(set(special_characters)):
            raise ValueError(f"Invalid name: {self.full_name}")

    def _checks(self) -> None:
        self._check_names()
        self._check_phone_number()

    def exists(self):
        return bool(self.db_instance)

    def delete(self)-> List[int]:
        if self.exists():
            return User.DB.remove(doc_ids=[self.db_instance.doc_id])
        return []
            

    def save(self, validate_data: bool = False) -> int:
        if validate_data:            
            self._checks()
            
        if self.exists():
            return -1
        else:
            return User.DB.insert(self.__dict__)


def get_all_users():
    return [User(**user) for user in User.DB.all()]


if __name__ == "__main__":
    from faker import Faker

    fake = Faker(locale="fr_FR")
    for _ in range(10):
        user = User(
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            phone_number=fake.phone_number(),
            address=fake.address(),
        )
        print(user.save(validate_data=True))

    print(get_all_users())
    nicole = User('Margaux', 'Humbert')
    print(nicole.db_instance)
    print(nicole.exists())
    print(nicole.delete())
