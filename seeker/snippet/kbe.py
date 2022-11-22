#date: 2022-11-22T17:08:47Z
#url: https://api.github.com/gists/61a0f2f8f7f9671721407e786efafb58
#owner: https://api.github.com/users/MikeNGarrett

# Forked: https://git.zapashcanon.fr/zapashcanon/kbe/
#To export a chat from the ZorkInc team, run:
# ./kbe.py ZorkInc
# It will create a folder ZorkInc in which you'll find a JSON file containing raw logs, a .log file containing human-readable export of the chat. It'll also download all attachments of the chat and put them in that same folder.
# Note: defaults to "general" channel in your team. Change references to this channel below under "topic_name" (3).
#!/usr/bin/python3

import json
import sys
import os
from datetime import datetime

conv_name = sys.argv[1]
conv_dir = "./" + conv_name
os.mkdir(conv_dir)

# Defaults to the last 1 million messages.
pg = 1000000

query = '{"method": "read", "params": {"options": {"channel": {"name": "'+ conv_name +'", "members_type": "team", "topic_name": "general", "pagination": {"num": ' + str(pg) + '}}}}}'

json_out = conv_dir + "/out.json"
log_out = conv_dir + "/conv.log"

def run_query(q):
    cmd = "keybase chat api -m '" + query + "' >" + json_out
    os.system(cmd)

run_query(query)

query_dl = []

def get_content_type(entry):
    return entry["msg"]["content"]["type"]

def get_sender(entry):
    return entry["msg"]["sender"]["username"]

def get_msg_id(entry):
    return str(entry["msg"]["id"])

def get_filename(entry):
    ctype = get_content_type(entry)
    if ctype == "attachment":
        return entry["msg"]["content"]["attachment"]["object"]["filename"]
    elif ctype == "attachmentuploaded":
        return entry["msg"]["content"]["attachment_uploaded"]["object"]["filename"]
    else:
        print("don't know how to get filename")
        exit(1)

def mk_out_filename(entry):
    return conv_dir + "/msg_id_" + get_msg_id(entry) + "_" + get_filename(entry)

msg_stack = list()

def outputmsgs():
    with open(json_out, 'r') as f:
        outputmsgs.json_data = json.load(f)
    for entry in outputmsgs.json_data["result"]["messages"]:
        out = ""
        ctype = get_content_type(entry)
        mid = get_msg_id(entry)
        content = entry["msg"]["content"]
        sent_at = entry["msg"]["sent_at"]
        if ctype == "text":
            out = "<" + get_sender(entry) + "> " + content["text"]["body"]
        elif ctype == "reaction":
            out = "* " + get_sender(entry) + ": " + content["reaction"]["b"]
        elif ctype == "attachment":
            file_name = mk_out_filename(entry)
            out = get_sender(entry) + " sent attachment " + file_name
            query = '{"method": "download", "params": {"options": {"channel": {"name": "' + conv_name + '", "members_type": "team", "topic_name": "general"}, "message_id": ' + mid + ', "output": "' + file_name + '"}}}'
            query_dl.append((file_name, query))
        elif ctype == "attachmentuploaded":
            out = get_sender(entry) + " attachment " + mk_out_filename(entry) + " uploaded"
        elif ctype == "edit":
            edit = content["edit"]
            out = get_sender(entry) + " edited message with id " + str(edit["messageID"]) + " to: " + edit["body"]
        elif ctype == "delete":
            out = get_sender(entry) + " deleted message with ids " + str(content["delete"]["messageIDs"])
        elif ctype == "unfurl":
            out = get_sender(entry) + " sent unfurl: " + str(content["unfurl"]["unfurl"]["url"])
        else:
            out = "(unknown message type '" + ctype + "')"
            # print(entry)
        msg_stack.append("#" + mid + " - " + datetime.utcfromtimestamp(sent_at).strftime('%Y-%m-%d %H:%M:%S') + " - " + out + '\n')
    res = not 'last' in outputmsgs.json_data["result"]["pagination"]
    if res:
        outputmsgs.next = outputmsgs.json_data["result"]["pagination"]["next"]
    return res

print("exporting messages...")

while outputmsgs():
    query = '{"method": "read", "params": {"options": {"channel": {"name": "' + conv_name + '", "members_type": "team", "topic_name": "general"}, "pagination": {"next": "' + outputmsgs.next + '", "num": ' + str(pg) + '}}}}'
    run_query(query)

with open(log_out, 'a') as outfile:
    while msg_stack:
        msg = msg_stack.pop()
        outfile.write(msg)

print("downloading attachments...")

for (f, q) in query_dl:
    print("downloading " + f)
    cmd = "keybase chat api -m '" + q + "' > /dev/null"
    os.system(cmd)
