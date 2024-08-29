#date: 2024-08-29T16:52:07Z
#url: https://api.github.com/gists/5290fa55e63c46023f99260af082a329
#owner: https://api.github.com/users/docsallover

from flask import render_template
from flask_sqlalchemy import Pagination

@app.route('/users')
def users():
    page = request.args.get('page', 1, type=int)
    pagination = User.query.paginate(page, per_page=10)
    return render_template('users.html', pagination=pagination)