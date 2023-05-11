#date: 2023-05-11T16:58:56Z
#url: https://api.github.com/gists/4387bed46327b4e97adfe5481b3f5037
#owner: https://api.github.com/users/masonsalma

import pyrebase

# configure Firebase
firebase_config = {
  "apiKey": "AIzaSyA4eZFrlL_iG9AhwCWmoT1p_nDQdAwEYUw",
  "authDomain": "saferheart-c5f98.firebaseapp.com",
  "databaseURL": "https://saferheart-c5f98-default-rtdb.firebaseio.com",
  "storageBucket": "saferheart-c5f98.appspot.com"
}

# initialize Firebase app
firebase = pyrebase.initialize_app(firebase_config)

# get a reference to the database
db = firebase.database()

# retrieve data from the database
data = db.child("SAFERHEART​").get()

# print the data
print(data.val())
