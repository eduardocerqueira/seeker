#date: 2022-05-10T16:58:46Z
#url: https://api.github.com/gists/610074aa35c7da81623301edfc38f63b
#owner: https://api.github.com/users/Cyber-Aku

# pip install instapi

from instapi import Client

username = "ACCNT_USERNAME"
password = "ACCNT_PASSWORD"

api = Client(username, password)

"""
Contacts JSON Template:

contacts = [
{
    "first_name": "obinet",
    "last_name": "tubbin",
    "phone_numbers":[],
    "email_addresses":["rstubbin0@gmail.com"]
},{
    "first_name": "orabel",
    "last_name": "oissieux",
    "phone_numbers":["4141062538"],
    "email_addresses":[]
}]
"""

contacts = []

print(api.link(contacts))