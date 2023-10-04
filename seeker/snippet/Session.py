#date: 2023-10-04T17:05:37Z
#url: https://api.github.com/gists/18557f4a3e28974ca345e9752f7fed71
#owner: https://api.github.com/users/stefanpejcic

from flask import Flask, session
from flask_session import Session



app = Flask(__name__)

# Configure session type and secret key
app.config['SESSION_TYPE'] = 'filesystem'  # You can use other options like 'redis' for better performance
app.config['SESSION_PERMANENT'] = True  # Session lasts until the browser is closed
app.config['SESSION_USE_SIGNER'] = True   # Sign the session cookie for security
app.config['SESSION_KEY_PREFIX'] = 'your_prefix_here'  # Customize the session key prefix
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lasts for 7 days

# Initialize the session extension
Session(app)
