#date: 2025-05-07T16:44:28Z
#url: https://api.github.com/gists/ff361cf098ded4c7f8e4a11f17d8b858
#owner: https://api.github.com/users/amnuts

import dateutil.parser
from rich.table import Table


def table_to_dict(rich_table: Table, transposed: bool = True) -> dict|list:
    """
    Convert a rich.Table into dict

    Args:
        rich_table (Table): A rich Table that should be populated by the DataFrame values
        transposed (bool): If True, the table is transposed (list of objects), otherwise it is a dict of lists

    Returns:
        Union[List[dict[str, Any]], dict[str, list[Any]]]: The data extracted from the Table
    """
    data = {x.header: [y for y in x.cells] for x in rich_table.columns}

    if not transposed:
        return data

    keys = list(data.keys())
    values = zip(*[data[key] for key in keys])
    return [dict(zip(keys, row)) for row in values]


def table_to_json(rich_table: Table) -> str:
    import json

    return json.dumps(table_to_dict(rich_table), indent=4)


def table_to_csv(rich_table: Table) -> str:
    """
    Convert a rich.Table into CSV format

    Args:
        rich_table (Table): A rich Table that should be populated by the DataFrame values

    Returns:
        str: A CSV string with the Table data as its values
    """
    import csv
    from io import StringIO

    data = table_to_dict(rich_table)
    headers = data[0].keys() if len(data) > 0 else []

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in data:
        writer.writerow(row.values())
    return output.getvalue()


def print_table(rich_table: Table, output: str):
    """
    Print the rich table to the console or convert it to JSON or CSV format.

    Args:
        rich_table: The rich table to print or convert
        output: What to do with the table - print to console, convert to JSON or CSV
    """
    from rich import print as pprint

    if output == "json":
        pprint(table_to_json(rich_table))
    elif output == "csv":
        pprint(table_to_csv(rich_table))
    else:
        from rich.console import Console
        Console().print(rich_table)
