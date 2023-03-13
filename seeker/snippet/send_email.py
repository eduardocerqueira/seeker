#date: 2023-03-13T16:57:49Z
#url: https://api.github.com/gists/411982a4171fd40eb89bfbbd480a90a5
#owner: https://api.github.com/users/FilipRazek

import os

OAUTH_CLIENT_ID = os.environ.get('OAUTH_CLIENT_ID')
OAUTH_CLIENT_SECRET = "**********"
APP_NAME = "**********"

def send_email(email, subject, body):
    oauth_token = "**********"
    credentials = "**********"
                                    None, None, "https: "**********"
    service = build('gmail', 'v1', credentials=credentials)
    message = build_message(GMAIL_USERNAME, email, subject, body)

    create_message = {'raw': base64.urlsafe_b64encode(
        message.as_bytes()).decode()}

    service.users().messages().send(userId="me", body=create_message).execute() create_message = {'raw': base64.urlsafe_b64encode(
        message.as_bytes()).decode()}

    service.users().messages().send(userId="me", body=create_message).execute()