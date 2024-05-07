#date: 2024-05-07T16:50:13Z
#url: https://api.github.com/gists/c66d4063e870e094088dacea5d63b0be
#owner: https://api.github.com/users/gj84

import re
import json

from vladiate import Vlad
from vladiate.validators import (
    Validator, ValidationException,
    UniqueValidator, 
    SetValidator,
    FloatValidator,
    IntValidator,
    Ignore,
    RangeValidator,
    RegexValidator,
    )
from vladiate.inputs import LocalFile


csv_file = "religion.csv"
state = "California"
country = "United States of America"
group = "religion"

url_regex = '^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?'

phone_regex = "^[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*$"


class JsonStringValidator(Validator):

    def __init__(self, **kwargs):
        super(JsonStringValidator, self).__init__(**kwargs)
        self.invalid_set = set([])


    def validate(self, field, row={}):
        if field == "" and self.empty_ok:
            return
        try:
            json.loads(field)

        except json.decoder.JSONDecodeError:
            self.invalid_set.add(field)
            raise ValidationException(
                f"'{field}' is not a valid JSON string"                
            )
        
    @property
    def bad(self):
        return self.invalid_set

    

class ItrekerCSVValidator(Vlad):
    source = LocalFile(csv_file)
    validators = {
        "name": [
            UniqueValidator()
            ],
        "website": [
            RegexValidator(url_regex, empty_ok=True)
            ],
        "categories": [Ignore()],
        "group": [
            SetValidator([group])
            ],
        "category": [Ignore()],
        "phone": [
            RegexValidator(phone_regex, empty_ok=True)
            ],
        "full_address": [Ignore()],
        "address_street": [Ignore()],
        "address_city": [Ignore()],
        "address_postcode": [
            IntValidator(empty_ok=True)
            ],
        "address_state": [
            SetValidator([state])
            ],
        "address_country": [
            SetValidator([country])
            ],
        "lat": [
            FloatValidator(),
            RangeValidator(low=-90, high=90)
            ],
        "long": [
            FloatValidator(),
            RangeValidator(low=-180, high=180)
            ],
        "open_hours": [
            JsonStringValidator(empty_ok=True)
            ],
        "time_zone": [Ignore()],
        "rating": [
            FloatValidator(empty_ok=True)
            ],
        "photos_array": [
            RegexValidator(url_regex, empty_ok=True)
        ],
        "street_view": [
            RegexValidator(url_regex, empty_ok=True)
        ],
        "business_status": [Ignore()],
        "about": [Ignore()],
        "logo": [
            RegexValidator(url_regex, empty_ok=True)
        ],
        "reservation": [Ignore()],
        "booking": [Ignore()],
        "menu": [Ignore()],
        "ordering": [Ignore()]
    }
