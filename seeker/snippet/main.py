#date: 2021-11-15T16:54:30Z
#url: https://api.github.com/gists/ed7cb17f5da0150b711d78eb457ae43d
#owner: https://api.github.com/users/mypy-play

from typing import Optional, List, Dict, Set, Tuple
import math


class Package:
    def __init__(self, amount: int, price: int, expiry: str):
        self.amount = amount
        self.price = price
        self.expiry = expiry


class Movement:
    def __init__(self, item: str, amount: int, price: int, tag: str):
        self.item = item
        self.amount = amount
        self.price = price
        self.tag = tag


class Warehouse:
    def __init__(self, inventory: Optional[Dict[str, List['Package']]] = None,
                 history: Optional[List['Movement']] = None):
        if inventory is None:
            self.inventory = {}
        else:
            self.inventory = inventory

        if history is None:
            self.history = []
        else:
            self.history = history

    def store(self, item: str, amount: int, price: int, expiry: str, tag: str)\
            -> None:

        if item in self.inventory:
            self.inventory[item] = [Package(amount, price, expiry)] +\
                    self.inventory[item]
            self.history.append(Movement(item, amount, price, tag))
            self.inventory[item] = sorted(self.inventory[item],
                                          key=lambda item: item.expiry,
                                          reverse=True)

        else:
            self.inventory.update({item: [Package(amount, price, expiry)]})
            self.history.append(Movement(item, amount, price, tag))

    def remove_expired(self, today: str) -> List['Package']:
        to_remove = []
        return_val = []

        for item, pkgs in self.inventory.items():
            for pkg in pkgs:
                if pkg.expiry < today:
                    to_remove.append(pkg)
                    return_val.append(pkg)
                    self.history.append(
                        Movement(item, -(pkg.amount), pkg.price, "EXPIRED"))

        for item, pkgs in self.inventory.items():
            for element in to_remove:
                if element in pkgs:
                    pkgs.remove(element)

        return return_val

#     def find_inconsistencies(self) -> Set[Optional[Tuple[str, int, int]]]:
# 
#         items_amount = {}
#         res = []
# 
#         for item, pkgs in self.inventory.items():
#             sum_list = []
#             for pkg in pkgs:
#                 sum_list.append(pkg.amount)
#             items_amount.update({item:sum(sum_list)})
# 
#         print(items_amount)
#         for elem in self.history:
#             items_amount[elem.item] -= elem.amount
# 
#         print(items_amount)
#         for item in items_amount:
#             if item != 0:
#                 res.append((item, item[0], 0))
#         print(set(res))
#         return res

    def average_prices(self) -> Dict[str, float]:
        result = {}
        for item, pkgs in self.inventory.items():
            averages = []
            suma = 0
            for pkg in pkgs:
                suma += pkg.amount
                averages.append(pkg.price*pkg.amount)
            av = sum(averages)/suma
            result.update({item: av})

        return result

    def best_suppliers(self) -> Set[str]:
        res: Dict[str, List[List[object]]] = {}
        result = []
        dict_of: Dict[str, List[object]] = {}
        list_of = []

        for mov in self.history:
            if mov.tag in dict_of:
                dict_of[mov.tag][0] += mov.amount
            else:
                dict_of.update({mov.tag: [mov.amount, mov.item]})
        
        for seller, commodity in dict_of.items():
            list_of.append([seller, commodity[0], commodity[1]])
        
        for key in self.inventory.keys():
            res.update({key:[]})
            for elem in list_of:
                if elem[2] == key:
                    res[key].append(elem)
                    res[key] = sorted(res[key], key=lambda x: x[1], reverse=True)

        for elem in res.values():
            result.append(elem[0][0])
            n = 1
            while n < len(elem):
                if elem[n][1] == elem[0][1]:
                    result.append(elem[n][0])
                n += 1

        return set(result)
                

def print_warehouse(warehouse: Warehouse) -> None:
    print("===== INVENTORY =====", end="")
    for item, pkgs in warehouse.inventory.items():
        print(f"\n* Item: {item}")
        print("    amount  price  expiration date")
        print("  ---------------------------------")
        for pkg in pkgs:
            print(f"     {pkg.amount:4d}   {pkg.price:4d}     {pkg.expiry}")
    print("\n===== HISTORY ======")
    print("    item     amount  price   tag")
    print("-------------------------------------------")
    for mov in warehouse.history:
        print(f" {mov.item:^11}   {mov.amount:4d}   "
              f"{mov.price:4d}   {mov.tag}")


def example_warehouse() -> Warehouse:
    wh = Warehouse()

    wh.store("rice", 100, 17, "20220202", "ACME Rice Ltd.")
    wh.store("corn", 70, 15, "20220315", "UniCORN & co.")
    wh.store("rice", 200, 158, "20771023", "RICE Unlimited")
    wh.store("peas", 9774, 1, "20220921", "G. P. a C.")
    wh.store("rice", 90, 14, "20220202", "Theorem's Rice")
    wh.store("peas", 64, 7, "20211101", "Discount Peas")
    wh.store("rice", 42, 9, "20211111", "ACME Rice Ltd.")

    return wh


def test1() -> None:
    wh = example_warehouse()

    for item, length in ('rice', 4), ('peas', 2), ('corn', 1):
        assert item in wh.inventory
        assert len(wh.inventory[item]) == length

    assert len(wh.history) == 7

    # uncomment to visually check the output:
    # print_warehouse(wh)


# def test2() -> None:
#     wh = example_warehouse()
#     assert wh.find_inconsistencies() == set()
# 
#     wh.inventory['peas'][0].amount = 9773
#     wh.history[4].price = 12
# 
#     assert wh.find_inconsistencies() == {
#         ('peas', 1, -1),
#         ('rice', 14, 90),
#         ('rice', 12, -90),
#     }


def test3() -> None:
    wh = example_warehouse()
    bad_peas = wh.inventory['peas'][-1]
    assert wh.remove_expired('20211111') == [bad_peas]
    assert len(wh.history) == 8

    mov = wh.history[-1]
    assert mov.item == 'peas'
    assert mov.amount == -64
    assert mov.price == 7
    assert mov.tag == 'EXPIRED'

    assert len(wh.inventory['peas']) == 1
    print_warehouse(wh)


# def test4() -> None:
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 9, 'Pear Shop') == (42, 42 * 9)
#     assert len(wh.history) == 8
#     assert wh.find_inconsistencies() == set()
# 
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 12, 'Pear Shop') \
#         == (42 + 25, 42 * 9 + 25 * 17)
#     assert len(wh.history) == 9
#     assert wh.find_inconsistencies() == set()
# 
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 14, 'Pear Shop') \
#         == (42 + 70, 42 * 9 + 70 * 17)
#     assert len(wh.history) == 9
#     assert wh.find_inconsistencies() == set()
# 
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 15, 'Pear Shop') \
#         == (42 + 100 + 90, 42 * 9 + 100 * 17 + 90 * 14)
#     assert len(wh.history) == 10
#     assert wh.find_inconsistencies() == set()
# 
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 16, 'Pear Shop') \
#         == (42 + 100 + 90 + 2, 42 * 9 + 100 * 17 + 90 * 14 + 2 * 158)
#     assert len(wh.history) == 11
#     assert wh.find_inconsistencies() == set()
# 
#     # uncomment to visually check the output:
#     # print_warehouse(wh)
# 
#     wh = example_warehouse()
#     assert wh.try_sell('rice', 500, 81, 'Pear Shop') \
#         == (42 + 100 + 90 + 200, 42 * 9 + 100 * 17 + 90 * 14 + 200 * 158)
#     assert len(wh.history) == 11
#     assert wh.find_inconsistencies() == set()


def test5() -> None:
    wh = example_warehouse()

    expected = {
        'rice': 80.875,
        'corn': 15,
        'peas': (9774 + 64 * 7) / (9774 + 64),
    }

    avg_prices = wh.average_prices()

    assert expected.keys() == avg_prices.keys()

#    for item in avg_prices:
#        assert math.isclose(avg_prices[item], expected[item])

    assert wh.best_suppliers() \
        == {'UniCORN & co.', 'G. P. a C.', 'RICE Unlimited'}


if __name__ == '__main__':
    test1()
#    test2()
    test3()
#    test4()
    test5()

