#date: 2022-03-07T17:00:31Z
#url: https://api.github.com/gists/f4385f098dbcb513ae70d72dcc3a1b7b
#owner: https://api.github.com/users/DanyF-github

@app.route('/admin')
def admin():
   return render_template('admin.html')


@app.route('/join')
def join():
   return render_template('join.html')