#date: 2023-03-15T17:03:07Z
#url: https://api.github.com/gists/1775ea581384bf4c56625e63cf035ff1
#owner: https://api.github.com/users/KeplerDE

class Bill:
    """
    Objects that contains data about of bill, such as total amount
    and period of the bill
    """

    def __init__(self, amount, period):        # ALT + Enter
        self.amount = amount
        self.period = period



class Flatmate:
    """
    Creates a flatmate person who lives in the flat
    and pays a share of the bill

    """
    def __init__(self, name, days_in_house):
        self.name = name
        self.days_in_house = days_in_house

    def pays(self, bill):
        pass

class PdfReport:
    """
    Create PDF File that contains data about
    the flatmates such as their names, their due amounts
    and the period of the bill

    """

    def __init__(self, filename):
        self.filename = filename

    def generate(self, flatmate1, flatmate2, bill)
        pass


the_bill = Bill(amount = 120, period = "March 2023")
john = Flatmate(name="John", days_in_house=20)
marry = Flatmate(name="Marry", days_in_house=25)
garry = Flatmate(name="Garry", days_in_house=24)


print(john.pays(bill=the_bill))
    








