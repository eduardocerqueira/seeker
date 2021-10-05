#date: 2021-10-05T16:55:03Z
#url: https://api.github.com/gists/1935e6d7e0481c7186465b775f4fb532
#owner: https://api.github.com/users/21187

from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Hello. I am alive!"

def run():
  app.run(host='0.0.0.0',port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()