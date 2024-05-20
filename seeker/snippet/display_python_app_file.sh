#date: 2024-05-20T17:11:31Z
#url: https://api.github.com/gists/e04f6b093085f7d1c5ca38aa19ebb27e
#owner: https://api.github.com/users/sylvainkalache

$ cat my-app.py
from flask import Flask, request, render_template
import gunicorn
import platform
import subprocess

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!\n" + "Python version: " + platform.python_version() + "\n"
