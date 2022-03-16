#date: 2022-03-16T16:52:08Z
#url: https://api.github.com/gists/9653aab336e598a06fb900b91ba6a212
#owner: https://api.github.com/users/vikram-sridhar-ml

from datetime import datetime
from pydantic import BaseModel

from typing import (
	List
)

class OrderDetails(BaseModel):
	OrderId: str
	OrderDate: datetime

class order(BaseModel):
	CustomerId :  int
	Orderdetails: List[OrderDetails]