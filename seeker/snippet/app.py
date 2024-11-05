#date: 2024-11-05T17:06:16Z
#url: https://api.github.com/gists/37645f39d48067e7f5f2ebc903ca9905
#owner: https://api.github.com/users/KashapovK

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
   return "Hello from WSGI server!"

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=8082)
