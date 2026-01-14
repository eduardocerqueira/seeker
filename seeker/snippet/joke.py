#date: 2026-01-14T17:09:52Z
#url: https://api.github.com/gists/861321548b2ed2906c24e72e434a4635
#owner: https://api.github.com/users/vrnico

import requests

# The API endpoint (where we ask for data)
url = "https://official-joke-api.appspot.com/random_joke"

# Send a GET request to the API
response = requests.get(url)

# Convert the response to Python data (dictionary)
data = response.json()

# Pull out the parts we care about
setup = data["setup"]
punchline = data["punchline"]

# Display the joke
print("Here is a random joke:\n")
print(setup)
print(punchline)