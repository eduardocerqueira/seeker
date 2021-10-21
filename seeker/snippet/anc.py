#date: 2021-10-21T16:53:05Z
#url: https://api.github.com/gists/eb4bcb434a4e3bfae84a4f00f59c5572
#owner: https://api.github.com/users/heaphyg

class Person():
    """
    Represents a Person with first and last names and an email address
    """
    def __init__(self, first_name, last_name, email):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.parents = list()

    def __repr__(self):
        return self.first_name


def find_all_ancestors(person_list):
    """
    Return dictionary with the form: {person: ancestor_list}
    Ancestors are the parents of the person plus any ancestors
    of the parents.
    Ie:
    gpa = Person("Gpa", "Smith", "gpa@smith.com")
    alan = Person("Alan", "Smith", "fred@smith.com")
    betty = Person("Betty", "Smith", "betty@smith.com")
    betty.parents = [alan]
    alan.parents = [gpa]
    result = find_all_ancestors([gpa, alan, betty])
    
    expected_result = {
        gpa: [],
        alan: [gpa],
        betty: [alan, gpa],
    }
    
    Mention implementation notes or assumptions here:
    """
    pass
    # Add code here, feel free to modify the Person class or add helper functions