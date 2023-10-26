#date: 2023-10-26T17:05:23Z
#url: https://api.github.com/gists/ed6eaaee73887984e629a96bf22c1f4d
#owner: https://api.github.com/users/FluffyDietEngine

from pydantic import BaseModel
from datetime import datetime
from time import sleep

class TestClass(BaseModel):
    attribute_1: str
    timestamp: datetime = datetime.now()


for i in range(5):
    test = TestClass(
        attribute_1="a"
    )
    print(test)
    sleep(5)