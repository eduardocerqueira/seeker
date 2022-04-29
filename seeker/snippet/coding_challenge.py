#date: 2022-04-29T17:04:05Z
#url: https://api.github.com/gists/ea58842f2090f99ddb1a29adb038b218
#owner: https://api.github.com/users/jdow

#!/usr/bin/env python

def main():
data = [
    ("uid=joe,ou=users,dc=acme,dc=com", {"uid": "joe", "firstName": "Joe", "LastName": "Jones", "department": "Sales", "location": "US"}),
    ("uid=john,ou=users,dc=acme,dc=com", {"uid": "john", "firstName": "John", "LastName": "Jacobson", "department": "Sales", "location": "CA"}),
    ("uid=joel,ou=users,dc=acme,dc=com", {"uid": "joel", "firstName": "Joel", "LastName": "Johnson", "department": "Marketing", "location": "US"}),
    ("uid=jack,ou=users,dc=acme,dc=com", {"uid": "jack", "firstName": "Jack", "LastName": "Jackson", "department": "Sales", "location": "US"}),
    ("uid=jason,ou=users,dc=acme,dc=com", {"uid": "jason", "firstName": "Jason", "LastName": "Jameson", "department": "Sales", "location": "CA"}),
    ("uid=jeremy,ou=users,dc=acme,dc=com", {"uid": "jason", "firstName": "Jeremy", "LastName": "Jordan", "department": "Marketing", "location": "CA"}),
    ("uid=jacob,ou=users,dc=acme,dc=com", {"uid": "jason", "firstName": "Jacob", "LastName": "Jenkins", "department": "Marketing", "location": "US"}),
]
# Write a function that returns a list of all people based in a specific location
# Write a function to returns a list of all people in a specific department
# Use those functions to find a list of all CA-based Sales people and print their first and last name
# Use those functions to find a list of all US-based Marketing people and print their UID

if __name__ = "__main__":
    main()