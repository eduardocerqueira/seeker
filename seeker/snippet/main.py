#date: 2022-03-28T17:15:06Z
#url: https://api.github.com/gists/a1d4e290e27823a5fa2f61548610a999
#owner: https://api.github.com/users/kushagrakumarovgu

import random
import string
from dataclasses import dataclass, field


def generate_id() -> str:
    return "".join(random.choices(string.ascii_uppercase,k=12))

# if frozen is set to True, the data members cannot be changed after initialization.
# Making data immutable. Also, __post_init__ will throw error as well.
@dataclass(frozen=False)
class Person:
    #NOTE: dataclass decorator automatically generates
    # a. initializer (__init__) method
    # b. __repr__ method.
    # So, No need to define them.
    # Cons: It Abuses the concept of class variable
    # to represent the instance variable.

    name: str
    address: str
    active: bool = True # we can assign default values as well. 
    email_addresses: list[str] = field(default_factory=list) #throwing error , don't know why ?
    id: str = field(init=False,default_factory=generate_id) # default_factory = takes a function.
    #init=False meansid will not be part of initializer(__init__)
    _search_string: str = field(init=False,repr=False)
    # repr=False will show _search_string when the object will be
    # printed but it will be a internal data member of class.

    # Intiializes data member post initialization.
    # In this case we don't want to search string to be
    # initialized by the user, rather to be constructed with
    # the provided name and address.
    def __post_init__(self) -> None:
        self._search_string = f"{self.name} {self.address}"
    

    

def main() -> None:
    person = Person(name="Krishna",address="Gokul")
    # Below line will throw error as id is Not part of initializer.
    #p2 = Person(name="Ajay",address="123 street",active=False,id="super cool")
    print(person)

    # below will throw error if frozen=True
    #person.name = "Rajesh"
    

if __name__ == '__main__':
    main()


    

