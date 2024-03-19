#date: 2024-03-19T17:00:18Z
#url: https://api.github.com/gists/496ae39fd8a66ae8468a8797161fc1a7
#owner: https://api.github.com/users/irvingpop

import requests

# Your LEGACY Slack API token
SLACK_API_TOKEN = "**********"

def get_channel_list():
    """
    Retrieves a list of available channels using the Slack API.
    """
    url = "https://slack.com/api/conversations.list"
    headers = {
        "Authorization": "**********"
    }
    params = {
        "types": "private_channel"  # Include both public and private channels
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if data.get("ok"):
        channels = data.get("channels", [])
        return channels
    else:
        print(f"Error retrieving channel list: {data.get('error', 'Unknown error')}")

def invite_user_to_channel(channel_id, user_id):
    url = "https://slack.com/api/conversations.invite"
    headers = {
        "Authorization": "**********"
    }
    params = {
        "channel": channel_id,
        "users": user_id
    }

    response = requests.post(url, headers=headers, params=params)
    data = response.json()

    if data.get("ok"):
        print(f"Successfully invited users to channel {channel_id}")
    else:
        print(f"Error inviting users: {data.get('error', 'Unknown error')}")

def join_channels(channel_list):
    """
    Joins the app to the specified channels.
    """
    for channel in channel_list:
        if channel['name'] == 'admins-cantina':
            continue

        print(f"Joining channel: {channel['name']}: {channel['id']}")
        invite_user_to_channel(channel['id'], "U06Q1PH2X40")


if __name__ == "__main__":
    channel_list = get_channel_list()
    if channel_list:
        join_channels(channel_list)
    else:
        print("No channels found.")
    else:
        print("No channels found.")
