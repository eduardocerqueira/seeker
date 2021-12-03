#date: 2021-12-03T17:15:18Z
#url: https://api.github.com/gists/bf5eba9339584dd31e1a2de0c99fd0e0
#owner: https://api.github.com/users/deknowny

from __future__ import annotations

import fastgql


@fastgql.type
class Order:
    cost: str = fastgql.field(
        fastgql.sql.SelectField("cost", of="order")
    )
    # Fetch owner of order
    owner: Person = fastgql.nested_field(
        rule=(
            fastgql.sql.Field("person.id")
            == fastgql.sql.Field("order.person_id")
        )
    )


@fastgql.type
class Person:
    name: str = fastgql.field(
        fastgql.sql.SelectField("name", of="person")
    )
    age: int = fastgql.field(
        fastgql.sql.SelectField("age", of="person")
    )
    # One line syntax
    # Fetch all orders to person
    all_orders: list[Order] = fastgql.nested_field(
        rule=(
            fastgql.sql.Field("person.id")
            == fastgql.sql.Field("order.person_id")
        )
    )
    # Or use function-like syntax (supports arguments)
    # Fetch last order for person
    last_order: Order = fastgql.nested_field(
        rule=(
            (
                fastgql.sql.Field("person.id") == fastgql.sql.Field("order.person_id")
            ) & (
                fastgql.sql.function("max", fastgql.sql.Field("order.person_id"))
            )
        )
    )
    

@fastgql.type
class Query:
    @fastgql.field(response_type=Person)
    def person(self, id: fastgql.ID):
        return fastgql.sql.Field("person.id") == fastgql.sql.Argument(id)
