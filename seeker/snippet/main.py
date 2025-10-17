#date: 2025-10-17T16:56:30Z
#url: https://api.github.com/gists/a1f5721a6f7900e4e14d9c076d51b776
#owner: https://api.github.com/users/mypy-play

from ib111 import week_04  # noqa

# V této úloze budete pracovat s databázovou tabulkou. Tabulka je
# dvojice složená z «hlavičky» a seznamu «záznamů». «Hlavička»
# obsahuje seznam názvů sloupců. Jeden záznam je tvořen seznamem
# hodnot pro jednotlivé sloupce tabulky (pro jednoduchost uvažujeme
# jenom hodnoty typu řetězec). Ne všechny hodnoty v záznamech musí
# být vyplněny – v tom případě mají hodnotu ‹None›.


# Vaším úkolem bude nyní otypovat a implementovat následující
# funkce. Funkce ‹get_header› vrátí hlavičku tabulky ‹table›.
table_ = tuple[list[str | None], list[list[str | None]]]
listt = list[str | None]


def get_header(table: table_) -> list[str]:
    list_of_header = []
    header, _ = table
    for h in header:
        list_of_header.append(h)
    return list_of_header

# Funkce ‹get_records› vrátí seznam záznamů z tabulky ‹table›.


def get_records(table: table_) -> list[list[str | None]]:
    list_of_records = []
    _, record = table
    for r in record:
        list_of_records.append(r)
    return list_of_records


# Procedura ‹add_record› přidá záznam ‹record› na konec tabulky
# ‹table›. Můžete předpokládat, že záznam ‹record› bude mít stejný
# počet sloupců jako tabulka.

def add_record(
                record: listt, table: table_
                ) -> list[list[str | None]]:
    _, records = table
    record_ = []
    for r in record:
        record_.append(r)
    records.append(record_)
    return records


# Predikát ‹is_complete› je pravdivý, neobsahuje-li tabulka ‹table›
# žádnou hodnotu ‹None›.

def is_complete(table: table_) -> bool:
    head, record = table
    y_or_n = 0
    for h in head:
        if h is None:
            y_or_n += 1
    for r in record:
        for r_ in r:
            if r_ is None:
                y_or_n += 1
    return y_or_n == 0

# Funkce ‹index_of_column› vrátí index sloupce se jménem ‹name›.
# Můžete předpokládat, že sloupec s jménem ‹name› se v tabulce
# nachází. První sloupec má index 0.


def index_of_column(name: str | None, header: list[str | None]) -> int:
    index_ = 0
    for n in header:
        if n != name:
            index_ += 1
        else:
            break
    return index_

# Funkce ‹values› vrátí seznam platných hodnot (tzn. takových, které
# nejsou ‹None›) v sloupci se jménem ‹name›. Můžete předpokládat, že
# sloupec se jménem ‹name› se v tabulce nachází.


def values(name: str, table: table_) -> list[str]:
    head, record = table
    index_h = 0
    list_of_correct_values = []
    for h in head:
        if h != name:
            index_h += 1
        else:
            break
    for r in record:
        index_r = 0
        for r_ in r:
            if index_h == index_r and r_ is not None:
                list_of_correct_values.append(r_)
            index_r += 1
    return list_of_correct_values


# Procedura ‹drop_column› smaže sloupec se jménem ‹name› z tabulky
# ‹table›. Můžete předpokládat, že sloupec se jménem ‹name› se
# v tabulce nachází.

def drop_column(name: str, table: table_) -> table_:
    head, record = table
    cop_head = head.copy()
    index_h = 0
    index_head = 0
    for _ in range(len(head)):
        head.pop()
    for h in cop_head:
        if h != name:
            head.append(h)
            index_h += 1
        else:
            index_head = index_h
    for r in record:
        cop_r = r.copy()
        for _ in range(len(r)):
            r.pop()
        index_r = 0
        for r_ in cop_r:
            if index_r != index_head:
                r.append(r_)
            index_r += 1
    return _

# Konečně otypujte následující dvě testovací funkce (jejich
# implementaci neměňte, pouze přidejte typové anotace).


def make_empty() -> tuple[list[str], list[None]]:
    return ["A", "B", "C", "D"], []


def make_table() -> table_:
    return (["A", "B", "C"],
            [["a1", "b1", None],
             ["a2", "b2", "c2"],
             ["a3", None, "c3"]])


def main() -> None:

    # header test
    assert get_header(make_empty()) == ['A', 'B', 'C', 'D']
    assert get_header(make_table()) == ['A', 'B', 'C']

    # records test
    assert get_records(make_empty()) == []
    assert get_records(make_table()) == [["a1", "b1", None],
                                         ["a2", "b2", "c2"],
                                         ["a3", None, "c3"]]

    # add_record test
    tab_1 = make_empty()
    add_record(["a", "b", "c", "d"], tab_1)
    assert tab_1 == (['A', 'B', 'C', 'D'], [['a', 'b', 'c', 'd']])

    tab_2 = make_table()
    add_record(["a4", None, None], tab_2)
    assert tab_2 == (['A', 'B', 'C'],
                     [['a1', 'b1', None], ['a2', 'b2', 'c2'],
                      ['a3', None, 'c3'], ['a4', None, None]])

    # is_complete test
    assert is_complete(make_empty())
    assert not is_complete(make_table())
    assert is_complete((["A", "B", "C"],
                        [["a1", "b1", "c1"], ["a2", "b2", "c2"]]))

    # index_of_column test
    header = ['A', 'C', 'B']
    assert index_of_column('A', header) == 0
    assert index_of_column('C', header) == 1
    assert index_of_column('B', header) == 2

    tab_v = make_table()
    assert values("A", tab_v) == ["a1", "a2", "a3"]
    assert values("B", tab_v) == ["b1", "b2"]
    assert values("C", tab_v) == ["c2", "c3"]
    assert values("B", make_empty()) == []

    # drop_column test
    tab_3 = make_table()
    drop_column("A", tab_3)
    assert tab_3 == (['B', 'C'],
                     [['b1', None], ['b2', 'c2'], [None, 'c3']])

    tab_4 = make_table()
    drop_column("B", tab_4)
    assert tab_4 == (['A', 'C'],
                     [['a1', None], ['a2', 'c2'], ['a3', 'c3']])

    tab_5 = make_empty()
    drop_column("D", tab_5)
    assert tab_5 == (['A', 'B', 'C'], [])


if __name__ == "__main__":
    main()
