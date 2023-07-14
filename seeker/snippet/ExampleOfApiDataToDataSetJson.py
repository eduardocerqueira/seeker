#date: 2023-07-14T16:47:10Z
#url: https://api.github.com/gists/0597c0a17d5a3a3d999453c017cdcc07
#owner: https://api.github.com/users/valor121

import requests
import json

# Send an HTTP GET request to the website to fetch the data
url1 = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=key'

url2 = 'https://newsapi.org/v2/top-headlines?country=se&apiKey=key' #key is a placeholder for the actual key

response = requests.get(url1, url2)

# Check if the request was successful
# status code 200 indicates success)
if response.status_code == 200:
    # Extract the data from the response
    data = response.json()

    # Create a dictionary to store the formatted API data
    api_data = {
        "intents": [
            {
                "tag": "news in english",
                "patterns": ["news in english"],
                "responses": [
                    {"name": result["source"]["name"], "description": result["description"]} for result in
                    data["articles"] if result["description"]
                ]
            },
            {
                "tag": "news in swedish",
                "patterns": ["news in swedish"],
                "responses": [
                    {"name": result["source"]["name"], "description": result["description"]} for result in
                    data["articles"] if result["description"] if ["language" == "se"]
                ]
            },


        ]
    }

       #save the formatted data as JSON in a file
    with open('api_data.json', 'w') as file:
        json.dump(api_data, file)
    print('Data saved successfully.')
else:
    print('Failed to retrieve data from the website.')


