#date: 2024-12-02T16:58:17Z
#url: https://api.github.com/gists/81ef5bca642ce98b6014ff87f0bbc11a
#owner: https://api.github.com/users/RobSpectre


import ldclient
from ldclient.config import Config
from ldclient import Context

from yourapp import App

app = App()

ldclient.set_config(Config('sdk-copy-your-key-here'))

client = ldclient.get()

@app.route("/")
def root():
    context = Context.builder('user-id-123abc').kind('user').name('Sandy').set('email', 'sandy@testcorp.com').build()   

    flag_value = client.variation("my-first-feature-flag", context, False)

    # If the flag is true, user gets new template. If not, fallback to previous template.
    if flag_value == True:
      return "new_template.html"
    else:
      return "old_template.html"