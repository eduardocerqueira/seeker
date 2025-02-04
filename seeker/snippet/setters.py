#date: 2025-02-04T16:54:47Z
#url: https://api.github.com/gists/4ab7d5fd72f5fef40e90b85756cdad6b
#owner: https://api.github.com/users/docsallover

def set_salary(self, salary):
    if salary < 0:
        raise ValueError("Salary cannot be negative")
    self.__salary = salary