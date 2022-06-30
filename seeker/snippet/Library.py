#date: 2022-06-30T21:22:19Z
#url: https://api.github.com/gists/a0341712bb857c4d4127edc24e9f10f8
#owner: https://api.github.com/users/edaily00

class LibraryItem:

    def __init__(self, library_item_id, title):
        self._library_item_id = library_item_id
        self._title = title
        self._location = "ON_SHELF"
        self._checked_out_by = "None"
        self._requested_by = "None"
        self._date_checked_out = 0

    def get_library_item_id(self):
        return self._library_item_id

    def get_location(self):
        return self._location

    def set_location(self, location):
        self._location = location

    def get_checked_out_by(self):
        return self._checked_out_by

    def set_checked_out_by(self, patron):
        self._checked_out_by = patron

    def get_requested_by(self):
        return self._requested_by

    def set_requested_by(self, patron):
        self._requested_by = patron

    def get_date_checked_out(self):
        return self._date_checked_out

    def set_date_checked_out(self, day):
        self._date_checked_out = day


class Book(LibraryItem):

    def __init__(self, library_item_id, title, author):
        super().__init__(library_item_id, title)
        self._author = author
        self._check_out_length = 21

    def get_check_out_length(self):
        return self._check_out_length


class Album(LibraryItem):

    def __init__(self, library_item_id, title, artist):
        super().__init__(library_item_id, title)
        self._artist = artist
        self._check_out_length = 14

    def get_check_out_length(self):
        return self._check_out_length


class Movie(LibraryItem):

    def __init__(self, library_item_id, title, director):
        super().__init__(library_item_id, title)
        self._director = director
        self._check_out_length = 7

    def get_check_out_length(self):
        return self._check_out_length


class Patron:

    def __init__(self, patron_id, patron_name):
        self._patron_id = patron_id
        self._patron_name = patron_name
        self._checked_out_items = []
        self._fine_amount = 0

    def get_patron_id(self):
        return self._patron_id

    def get_patron_name(self):
        return self._patron_name

    def get_fine_amount(self):
        return self._fine_amount / 100

    def get_checked_out_items(self):
        return self._checked_out_items

    def add_library_item(self, library_item):
        self._checked_out_items.append(library_item)

    def remove_library_item(self, library_item):
        self._checked_out_items.remove(library_item)

    def amend_fine(self, amount):
        if amount > 0:
            amount = int((amount * 100))
            self._fine_amount += amount
        else:
            amount = int((amount * 100))
            self._fine_amount -= abs(amount)


class Library:

    def __init__(self):
        self._holdings = {}
        self._members = {}
        self._current_date = 0

    def add_library_item(self, library_item):
        self._holdings[library_item.get_library_item_id()] = library_item

    def get_holdings(self):
        return self._holdings

    def add_patron(self, patron):
        self._members[patron.get_patron_id()] = patron

    def lookup_library_item_from_id(self, library_item_id):
        if library_item_id in self._holdings:
            return self._holdings[library_item_id]
        else:
            return

    def lookup_patron_from_id(self, patron_id):
        if patron_id in self._members:
            return self._members[patron_id]
        else:
            return

    def check_out_library_item(self, patron_id, library_item_id):
        if patron_id in self._members:
            if library_item_id in self._holdings:
                if self._holdings[library_item_id].get_checked_out_by() == "None":
                    if self._holdings[library_item_id].get_requested_by() == "None":
                        self._holdings[library_item_id].set_checked_out_by(patron_id)
                        self._holdings[library_item_id].set_date_checked_out(self._current_date)
                        self._holdings[library_item_id].set_location("CHECKED_OUT")
                        self._members[patron_id].add_library_item(self._holdings[library_item_id])
                    elif self._holdings[library_item_id].get_requested_by() == patron_id:
                        self._holdings[library_item_id].set_checked_out_by(patron_id)
                        self._holdings[library_item_id].set_date_checked_out(self._current_date)
                        self._holdings[library_item_id].set_location("CHECKED_OUT")
                        self._members[patron_id].add_library_item(self._holdings[library_item_id])
                        self._holdings[library_item_id].set_requested_by("None")
                    else:
                        return print("item on hold by other patron")
                else:
                    return print("item already checked out")
            else:
                return print("item not found")
        else:
            return print("patron no found")

        return print("check out successful")

    def return_library_item(self, library_item_id):
        if library_item_id in self._holdings:
            if self._holdings[library_item_id].get_location() != "ON_SHELF":
                self._members[self._holdings[library_item_id].get_checked_out_by()].remove_library_item(self._holdings[library_item_id])
                self._holdings[library_item_id].set_checked_out_by("None")
                if self._holdings[library_item_id].get_requested_by() == "None":
                    self._holdings[library_item_id].set_location("ON_SELF")
                elif self._holdings[library_item_id].get_requested_by() != "None":
                    self._holdings[library_item_id].set_location("ON_HOLD_SHELF")
            else:
                return print("item already in library")
        else:
            return print("item not found")

        return print("return successful")

    def request_library_item(self, patron_id, library_item_id):
        if patron_id in self._members:
            if library_item_id in self._holdings:
                if self._holdings[library_item_id].get_requested_by() == "None":
                    self._holdings[library_item_id].set_requested_by(patron_id)
                    self._holdings[library_item_id].set_location("ON_HOLD_SHELF")
                    return print("request successful")
                else:
                    return print("item already on hold")
            else:
                return print("item not found")
        else:
            return print("patron not found")

    def pay_fine(self, patron_id, amount):
        if patron_id in self._members:
            self._members[patron_id].amend_fine(amount)
            return print("payment successful")
        else:
            return print("patron not found")

    def increment_current_date(self):
        self._current_date += 1
        for member in self._members:
            for item in self.lookup_patron_from_id(member).get_checked_out_items():
                if self._current_date >= item.get_date_checked_out() + item.get_check_out_length():
                    self.lookup_patron_from_id(member).amend_fine(.10)


"""


lib = Library()

man = Patron(555, "Eric")
women = Patron(777, "Jess")

book1 = Book(111, "cool movie", "writer")
book2 = Book(333, "its a book", "writer")
movie1 = Movie(222, "movie", "Lord")
lib.add_library_item(book1)
lib.add_library_item(movie1)
lib.add_library_item(book2)
lib.add_patron(man)
lib.add_patron(women)
print(lib._current_date)
print(lib._current_date)

lib.check_out_library_item(555, 111)
lib.check_out_library_item(555, 333)


for _ in range(23):
    lib.increment_current_date()

print(man.get_fine_amount())
lib.pay_fine(555, -.4)
print(man.get_fine_amount())



for _ in range(21):
    lib.increment_current_date()
print(man.get_fine_amount())
man.amend_fine(-4.2)
print(man.get_fine_amount())

"""

"""for member in lib._members:
    for item in lib.lookup_patron_from_id(member).get_checked_out_items():
        if lib._current_date > item.get_date_checked_out() + item.get_check_out_length():
            lib.lookup_patron_from_id(member).amend_fine(.10)
"""


















