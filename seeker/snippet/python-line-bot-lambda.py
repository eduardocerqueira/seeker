#date: 2023-07-05T16:47:48Z
#url: https://api.github.com/gists/9a90526d7ef11c340a26182554255991
#owner: https://api.github.com/users/taikis

import os

from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    MessagingApi,
    TextMessage,
    ReplyMessageRequest,
    Configuration,
    ApiClient
    )
from linebot.v3.webhooks import (MessageEvent,TextMessageContent)

channel_access_token = "**********"
channel_secret = "**********"

configuration = "**********"
handler = "**********"

@handler.add(MessageEvent, message=TextMessageContent)
def message(line_event):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        print(line_bot_api)
        reply_text = line_event.message.text
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token= "**********"
                messages=[TextMessage(text=reply_text)]
            )
        )
        
def lambda_handler(event, context):
    signature = event["headers"]["x-line-signature"]
    body = event["body"]
    handler.handle(body, signature)signature = event["headers"]["x-line-signature"]
    body = event["body"]
    handler.handle(body, signature)