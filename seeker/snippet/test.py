#date: 2022-04-27T16:58:17Z
#url: https://api.github.com/gists/78f85b84bb07c8f1460fd0b51cd9b674
#owner: https://api.github.com/users/Woovie

import json

# pylint: disable-all
class ScanChannel:
    def __init__(self):
        self.user_history = {}
        self.last_message = 0
        self.channel = DiscordChannel('photos.json')# https://jsonplaceholder.typicode.com/photos
        self.channel.read()

    def scan_channel(self):
        start_message = None
        end_message = None
        reached_end = False
        while not reached_end:
            if end_message:
                start_message = end_message
            end_message = self.scan(self.channel.history(start_message))
            if end_message == start_message:
                reached_end = True

    def scan(self, messages: list):
        last_message = None
        for message in messages:
            user_id = message['albumId']# Pretend this is the User ID like Discord
            if not user_id in self.user_history:
                self.user_history[user_id] = 0
            self.user_history[user_id] += 1
            last_message = message['id']
        return last_message


class DiscordChannel:
    def __init__(self, filename: str):
        self.file = filename
        self.messages: list = None

    def read(self):
        with open(self.file, 'r') as rawdata:
            read_data = json.load(rawdata)
        read_data.reverse()# Ensure it's backwards like Discord
        self.messages = read_data

    def history(self, message_id: int = 0, limit: int = 100):
        if not message_id:
            message_id = 0
        if message_id > 0:
            message_id = self.find_message_by_id(message_id)
        return self.messages[message_id:message_id+limit]

    def find_message_by_id(self, id: int):
        for message in self.messages:
            if message['id'] == id:
                return self.messages.index(message)

def main():
    scanner = ScanChannel()
    scanner.scan_channel()
    print(scanner.user_history)


if __name__ == "__main__":
    main()