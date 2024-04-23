#date: 2024-04-23T16:59:50Z
#url: https://api.github.com/gists/eb28c85b9b38ec24ddd671be005612f3
#owner: https://api.github.com/users/xmarva

from flask import Flask
app = Flask(__name__)
 
@app.route('/')
def upload_file():
    return "Hey there!"

if __name__ == '__main__':
    app.run(debug=True)