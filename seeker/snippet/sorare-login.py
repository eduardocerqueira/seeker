#date: 2023-01-16T17:07:31Z
#url: https://api.github.com/gists/a750cf7af1fa757b4bae7b870db0f6c4
#owner: https://api.github.com/users/asilbalaban

import requests
import bcrypt

"""
pip3 install bcrypt
"""

def getSalt(email):
    url = "https://api.sorare.com/api/v1/users/" + email
    response = requests.get(url)
    return response.json()["salt"]

 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********"( "**********"h "**********"a "**********"s "**********"h "**********"e "**********"d "**********"P "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"e "**********"m "**********"a "**********"i "**********"l "**********", "**********"  "**********"a "**********"u "**********"d "**********") "**********": "**********"
    headers = {
        'content-type': 'application/json',
    }

    json_data = {
        'operationName': 'SignInMutation',
        'variables': {
            'input': {
                'email': email,
                'password': "**********"
            },
        },
        'query': "**********": signInInput!) { signIn(input: $input) { currentUser { slug jwtToken(aud: "'+aud+'") { token expiredAt } } errors { message } } }',
    }

    response = requests.post('https://api.sorare.com/graphql', headers=headers, json=json_data)
    return response.json()

if __name__ == "__main__":
    aud = "<your-aud>"
    email = "<your-email>"
    password = "**********"
    salt = getSalt(email).encode('utf-8')
    hashedPassword = "**********"
    response = "**********"
    print(response)
