#date: 2023-09-11T17:05:02Z
#url: https://api.github.com/gists/9aba0892afc51fd50859e331d6dd4c10
#owner: https://api.github.com/users/rfcantcode

from flask import Flask
import os, random, string, time

app = Flask(__name__)

def randstr():
    t=random.randint(0, 10)
    x=random.randint(8,256)
    print(f"[-]waiting {t} - length {x}")
    time.sleep(t)
    return ''.join(random.choice(string.ascii_letters) for i in range(x))

#default route
@app.route('/')
def index():
    url="""
        <a href="{randstr}">{randstr}</a>
    """.format(randstr=randstr())
    return url

#anything else
@app.route('/<string:x>')
def roundabout(x):
    url="""
        <a href="{randstr}">{randstr}</a>
    """.format(randstr=randstr())
    return url
