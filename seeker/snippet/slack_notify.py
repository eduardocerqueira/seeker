#date: 2022-12-01T17:09:33Z
#url: https://api.github.com/gists/709a2896529e712f03d656bf34b7a873
#owner: https://api.github.com/users/shreyasgm

"""
Decorator usage @slack_notify_decorator(msg_on_success, channel)

Requires environment variable SLACK_BOT_TOKEN to be set

Suggest running app as:
SLACK_BOT_TOKEN= "**********"

App setup instructions here:
https://api.slack.com/authentication/basics

To send a message to a private channel, add the bot to the channel as described here:
https://www.ibm.com/docs/en/z-chatops/1.1.0?topic=slack-adding-your-bot-user-your-channel
"""

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from functools import wraps

def slack_msg(msg, channel="code_status"):
    client = "**********"=os.environ['SLACK_BOT_TOKEN'])
    try:
        response = client.chat_postMessage(channel=channel, text=msg)
        assert response["message"]["text"] == msg
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")


def slack_notify_decorator(msg_on_success, channel="code_status"):
    """
    Send a slack message on function run

    Args:
        msg_on_success (required)
    """
    import time
    import datetime

    def inner_decorator(func):
        @wraps(func)
        def wrapper(*arg, **kw):
            try:
                t = time.time()
                res = func(*arg, **kw)
                runtime = str(datetime.timedelta(seconds=(time.time() - t)))
                timing_msg = f"TIMING: function {func.__name__} took {runtime}."
                success_msg = f"Success: {msg_on_success}\n{timing_msg}"
                slack_msg(success_msg, channel)
                return res
            except Exception as e:
                runtime = str(datetime.timedelta(seconds=(time.time() - t)))
                timing_msg = f"TIMING: function {func.__name__} ran for {runtime}."
                msg_on_fail = f"Failed :( Reason:\n{e}"
                fail_msg = f"{msg_on_fail}\n{timing_msg}"
                slack_msg(fail_msg, channel)
                raise

        return wrapper

    return inner_decorator
er_decorator
