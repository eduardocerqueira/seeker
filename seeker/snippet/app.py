#date: 2022-04-29T16:46:53Z
#url: https://api.github.com/gists/f8b74b66ff54b143b1606d9b55071d2a
#owner: https://api.github.com/users/vjohs

from flask import Flask

app = Flask(__name__)


@app.route('/')
def root():
    return '<h1>VJOHS</h1>'


if __name__ == '__main__':
    app.run(debug=True, port=4118)