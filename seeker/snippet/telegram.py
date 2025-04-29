#date: 2025-04-29T17:09:03Z
#url: https://api.github.com/gists/5de2710fd95d8e43c678512f967d6fd1
#owner: https://api.github.com/users/dolohow

from pyrogram import Client
from pyrogram.errors import FloodWait
import time

# Replace with your actual API credentials
API_ID = 
API_HASH = ""
BOT_TOKEN = "**********"

# The string you want to search for
FORBIDDEN_STRING = ""  # Replace with the string you're searching for

# Chat ID of the group or channel (can be a negative ID for private group/channel)
CHAT_ID = ""

app = Client("my_bot", api_id=API_ID, api_hash=API_HASH)

def delete_forbidden_messages():
    with app:
        # Get the chat history (limit=100 means it fetches the last 100 messages)
        messages = app.get_chat_history(CHAT_ID, limit=10000)

        # Iterate over the messages
        for message in messages:
            if FORBIDDEN_STRING in message.text:
                try:
                    app.delete_messages(CHAT_ID, message.id)
                    print(f"Deleted message: {message.text}")
                except FloodWait as e:
                    print(f"Rate limited, sleeping for {e.x} seconds.")
                    time.sleep(e.x)
                except Exception as e:
                    print(f"Error deleting message {message.id}: {e}")

def main():
    delete_forbidden_messages()

if __name__ == '__main__':
    main()
