#date: 2025-11-26T16:45:04Z
#url: https://api.github.com/gists/6e8bc86f3a9d99c87c39c950b94f73c1
#owner: https://api.github.com/users/laura-james

@web_site.route('/moviesadd',methods = ['GET', 'POST'])
def moviesadd():
  msg = ""
  if request.method == 'POST':
    name = request.form["name"]
    desc = request.form["descr"]
    year = request.form["year"]
    con = sqlite3.connect('movies.db')
    sql = "INSERT INTO Movies(name,description,year) VALUES(?,?,?)"
    cursor = con.cursor()
    cursor.execute(sql,(name,desc,year))
    con.commit()
    msg = name + " added to the Movies table"
  return render_template("addmovie.html",msg = msg)