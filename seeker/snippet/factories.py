#date: 2026-03-16T17:42:17Z
#url: https://api.github.com/gists/cd0f68e1b665833ec1fa02e2742787a0
#owner: https://api.github.com/users/96tm

import factory
from faker import Faker
from faker_optional import OptionalProvider

from app.models.part import PartModel
from app.schema.part import PartPublicSchema  # noqa:
from app.schema.part import PartUpdateSchema  # noqa:
from app.schema.part import PartCreateSchema

Faker.seed(0)  # reproducible random

fake = Faker()
fake.add_provider(OptionalProvider)


class PartModelFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = PartModel

    name = fake.word()
    description = fake.text()
    footprint = fake.bothify(text="SO?-#", letters="PT")
    manufacturer = fake.company()
    mpn = fake.bothify(text="???-####-###-???")
    notes = fake.text()


class PartCreateSchemaFactory(factory.Factory):
    class Meta:
        model = PartCreateSchema

    name = fake.word()
    description = fake.text()
    footprint = fake.bothify(text="SO?-#", letters="PT")
    manufacturer = fake.company()
    mpn = fake.bothify(text="???-####-###-???")
    notes = fake.text()
