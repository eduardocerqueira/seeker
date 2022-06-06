#date: 2022-06-06T17:09:54Z
#url: https://api.github.com/gists/ffb6bbe534736584e4f0f17e3bdc0b4d
#owner: https://api.github.com/users/amritshenava98

def isEmailFake(email):
    temp_mail = json.loads(open('utils/email.json').read())
    splitter = email.split("@")
    domain = splitter[1]
    for bogus_domain in temp_mail:
      if(domain == bogus_domain):
        return True
      else:
        return False