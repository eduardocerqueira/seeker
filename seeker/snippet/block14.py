#date: 2022-03-07T17:00:31Z
#url: https://api.github.com/gists/1081138465868886629ccfd2eefafbc1
#owner: https://api.github.com/users/DanyF-github

@app.route('/', methods=['POST', 'GET'])
def index():
   if request.method == 'POST':
       token = client.generate_token(session_id)
       admin = False
       if 'admin' in request.form:
           admin = True
       name = request.form['name']
       return render_template('index.html', session_id=session_id, token=token, is_admin=admin, name=name,
                              api_key=opentok_api)
   return 'please log in'