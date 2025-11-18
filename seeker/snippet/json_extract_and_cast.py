#date: 2025-11-18T17:15:20Z
#url: https://api.github.com/gists/ff09cdfe007a0380add58d9331b7523d
#owner: https://api.github.com/users/EnriqueSoria

from django.db.models import CharField
from django.db.models import Func
from django.db.models import Value
from django.db.models.functions import Cast


class JSONExtractAndCast(Cast):
    def __init__(self, expression, *paths, output_field):
        # borrowed from django_mysql.models.functions.JSONExtract
        exprs = [expression]
        for path in paths:
            if not hasattr(path, "resolve_expression"):
                path = Value(path)
            exprs.append(path)

        extracted_value = Func(
            *exprs,
            function="JSON_EXTRACT",
            output_field=CharField(),
        )
        unquoted_value = Func(
            extracted_value,
            function="JSON_UNQUOTE",
            output_field=output_field,
        )
        super().__init__(unquoted_value, output_field=output_field)