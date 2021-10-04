#date: 2021-10-04T17:16:34Z
#url: https://api.github.com/gists/ba51bbfbc27e1f195e049302afa34b4e
#owner: https://api.github.com/users/decay2Rn

#!/usr/bin/python3

# Python script to post multiple message to slack 
# Initially post files to slack without channel_id or any other details.
# Post main message with specific channel_id and files links in slack server.
# Following this method will add an 'edited' tag in the message.
# ENV variables SLACK_TOKEN & SLACK_CHANNEL needs to be added

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_token = os.environ.get("SLACK_TOKEN", "xoxb-xxxx") 
slack_channel = os.environ.get("SLACK_CHANNEL", "test-channel")
client = WebClient(token=slack_token)
        
def main():
    files = ['file1.png', 'file2.png'] # array contaning file names
    permalink = []
    try:
        for file in files:
            # upload files
            response = client.files_upload(file=file) 
            # get permalink from slack api response
            permalink.append(response['file']['permalink'])
        # append permalink with main messages.. only 2 files added here.
        msg = f"This is a test message with 2 files \n  <{permalink[0]}| {files[0]}> & <{permalink[1]}| {files[1]}>"
        # post main message
        response = client.chat_postMessage(
            channel=slack_channel,
            text=msg,
        )
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
            
if __name__ == "__main__":
    main()