#date: 2023-08-09T17:01:57Z
#url: https://api.github.com/gists/2a2f13ec265aab7cbee7f44f993e7be3
#owner: https://api.github.com/users/zishasch

import requests

# Replace with your Intercom Access Token
ACCESS_TOKEN = "**********"

# Function to send a message to the user
def send_message(user_id, message):
    url = f'https://api.intercom.io/conversations'
    headers = {
        'Authorization': "**********"
        'Content-Type': 'application/json'
    }
    payload = {
        'body': f'{message}',
        'from': {
            'type': 'user',
            'id': user_id
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        print('Message sent successfully!')
    else:
        print('Failed to send message.')

# Example user ID and message
user_id = 'USER_ID'
message = 'Hello, this is your chatbot. How can I assist you?'

# Call the function to send the message
send_message(user_id, message)
e(user_id, message)
