#date: 2024-08-29T16:56:54Z
#url: https://api.github.com/gists/f31e7d620f9525d0e6e7dd81e3d47be9
#owner: https://api.github.com/users/docsallover

from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('500.html'), 500