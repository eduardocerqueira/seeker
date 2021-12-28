#date: 2021-12-28T17:17:09Z
#url: https://api.github.com/gists/61e0e5632bac60e8d8c0016ed92da82d
#owner: https://api.github.com/users/llimllib

from rich.console import Console
from rich.style import Style
from rich.table import Table
from getchar import getchar

SELECTED = Style(bgcolor="white")


def print_table(console, table, rows=[], selected=0):
    for i, row in enumerate(rows):
        if i == selected:
            table.add_row(*row, style=SELECTED)
        else:
            table.add_row(*row)

    console.print(table)


def make_table():
    table = Table.grid(expand=True)

    table.add_column()
    table.add_column()
    table.add_column()

    return table


UP = "\x1b[A"
DOWN = "\x1b[B"
ENTER = "\r"

selected = 0
console = Console()

items = [
    ("a", "b", "c"),
    ("bb", "b", "c"),
    ("ccc", "b", "c"),
    ("dddd", "b", "c"),
]

console.clear()
print_table(console, make_table(), items, selected)

while True:
    ch = getchar(False)
    if ch == UP:
        selected = max(0, selected - 1)
    if ch == DOWN:
        selected = min(len(items) - 1, selected + 1)
    if ch == ENTER:
        print("you selected: ", items[selected])
        break
    console.clear()
    print_table(console, make_table(), items, selected)
