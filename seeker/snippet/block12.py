#date: 2022-03-07T17:00:31Z
#url: https://api.github.com/gists/d4b2c4b21233b02ea1ddd3a4d26e38dd
#owner: https://api.github.com/users/DanyF-github

@app.route('/test', methods=['POST', 'GET'])
def index():
    return "Hello, World!"