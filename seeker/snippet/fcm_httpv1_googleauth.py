#date: 2021-09-22T17:05:29Z
#url: https://api.github.com/gists/84377151578739b9ad286ff5ab04625b
#owner: https://api.github.com/users/lnstadrum

import google
import google.auth.transport.requests

def send_message_example(firebase_token):
    """ This sends a message using FCM to a specific Android device using google-auth library
          firebase_token: device-specific Firebase token (to get from the device itself)
    """
    # get creds
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        "json_key_file.json",
        scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    
    # obtain a temporary token to be able to send the requests
    credentials.refresh(google.auth.transport.requests.Request())
      # credentials could be kept (no need to reset them at every message)
      # refresh if credentials.expired == True

    # set up session
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    
    # prepare URL and JSON message to send
    url = "https://fcm.googleapis.com/v1/projects/{}/messages:send".format(credentials.project_id)
    msg = {
        "message": {
            "token": firebase_token,
            "data": {
                "arbitrary": "stuff"
            }
        }
    }

    # send the request
    resp = session.post(url, json=msg)

    # check the response
    if resp.status_code == 200:
        print('All good')
    else:
        print(f'Error sending wakeup message ({resp.status_code}): {resp.text}')