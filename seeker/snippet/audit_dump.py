#date: 2022-06-13T17:17:57Z
#url: https://api.github.com/gists/629feb883673894716f959b46e65a359
#owner: https://api.github.com/users/ThatGuyJustin

import requests
import json
from datetime import datetime
import time

DATE_FILE = datetime.now().strftime("%b_%d_%Y-%H_%M_%S")

# Insert Token to be able to query the Audit Log
# BOT_TOKEN = "INSERT TOKEN HERE"

# June 12th @ Noon EST
START_TIME = 1655049600000

# Guild ID
GID = 000000000000000000

headers = {
    "Authorization": f"Bot {BOT_TOKEN}",
    "User-Agent": "AuditLog-Dumper (https://github.com/ThatGuyJustin, 1.0)",
    "Content-Type": "application/json"
}

bare_records = {
    "application_commands": {},
    "audit_log_entries": {},
    "integrations": {},
    "threads": {},
    "users": {},
    "webhooks": {}
}


# Convert Snowflake to TimeStamp
def snowflake_to_timestamp(snowflake):
    return (snowflake >> 22) + 1420070400000

# Convert Snowflake to log string
def snowflake_to_timestring(snowflake):
    return datetime.utcfromtimestamp(snowflake_to_timestamp(snowflake) / 1e3).strftime("%b/%d/%Y %H:%M:%S")

# Section out actions into groups
USER_ACTIONS = [20, 22, 23, 24, 25]
CHANNEL_ACTIONS = [10, 11, 12, 13, 14, 15]
ROLE_ACTIONS = [30, 31, 32]

# Push entries to file
def generate_log_entries(data):
    with open(f"Output-FORMATTED-{DATE_FILE}.log", "w", encoding='utf-8') as output:

        for entry in data['audit_log_entries']:
            log_format = f"[{snowflake_to_timestring(int(entry['id']))}] "
            if entry['action_type'] in USER_ACTIONS:
                log_format += "[TYPE: USER UPDATE] "
                actor = None
                affected = None
                if entry['user_id'] in bare_records['users']:
                    actor = bare_records['users'][entry['user_id']]
                if entry['target_id'] in bare_records['users']:
                    affected = bare_records['users'][entry['target_id']]

                actions = {
                    20: "kicked",
                    22: "banned",
                    23: "removed ban for",
                    24: "updated",
                    25: "updated roles for"
                }
                log_format += "User "
                if actor:
                    log_format += f"{actor['username']}#{actor['discriminator']} ({entry['user_id']})"
                else:
                    log_format += entry['user_id']
                log_format += f" {actions.get(entry['action_type'])} "
                if affected:
                    log_format += f"{affected['username']}#{affected['discriminator']} ({entry['target_id']})"
                else:
                    log_format += entry['target_id']

                if entry['action_type'] in [20, 22, 23]:
                    if entry.get('reason'):
                        log_format += f"REASON: {entry['reason']}"
                if entry['action_type'] == 25:
                    action = "added" if entry['changes'][0]['key'] == "$add" else "removed"
                    log_format += f": {action} {entry['changes'][0]['new_value'][0]['name']} ({entry['changes'][0]['new_value'][0]['id']})"
                output.write(log_format + "\n")
            if entry['action_type'] in ROLE_ACTIONS:
                log_format += "[TYPE: ROLE UPDATE] "
                actor = None
                if entry['user_id'] in bare_records['users']:
                    actor = bare_records['users'][entry['user_id']]

                actions = {
                    30: "created role",
                    31: "updated role",
                    32: "deleted role",
                }

                log_format += "User "
                if actor:
                    log_format += f"{actor['username']}#{actor['discriminator']} ({entry['user_id']}) "
                else:
                    log_format += entry['user_id']
                log_format += f"{actions.get(entry['action_type'])} {entry['target_id']}: {entry['changes']}"
                output.write(log_format + "\n")
            if entry['action_type'] in CHANNEL_ACTIONS:
                log_format += "[TYPE: CHANNEL UPDATE] "
                actor = None
                if entry['user_id'] in bare_records['users']:
                    actor = bare_records['users'][entry['user_id']]

                actions = {
                    10: "created channel",
                    11: "updated channel",
                    12: "deleted channel",
                    13: "created channel permissions for",
                    14: "updated channel permissions for",
                    15: "deleted channel permissions for",
                }

                log_format += "User "
                if actor:
                    log_format += f"{actor['username']}#{actor['discriminator']} ({entry['user_id']}) "
                else:
                    log_format += entry['user_id']
                log_format += f"{actions.get(entry['action_type'])} {entry['target_id']}: {entry['changes']}"
                output.write(log_format + "\n")
            print(entry)
            print(log_format)
        output.close()


def main():
    url = f"https://discord.com/api/v10/guilds/{GID}/audit-logs?limit=100"

    last_id = None
    cycles = 0
    
    # Send request, then sleep for a second if we aren't done
    while True:
        # try:
        if last_id:
            url += f"&before={last_id}"
        r = requests.get(url, headers=headers)
        # Interate once through all the data, and make a "master copy" of all the unique keys to dump to json file
        for entry in r.json()['application_commands']:
            if not bare_records['application_commands'].get(entry['id']):
                bare_records['application_commands'][entry['id']] = entry
        for entry in r.json()['audit_log_entries']:
            if not bare_records['audit_log_entries'].get(entry['id']):
                bare_records['audit_log_entries'][entry['id']] = entry
        for entry in r.json()['integrations']:
            if not bare_records['integrations'].get(entry['id']):
                bare_records['integrations'][entry['id']] = entry
        for entry in r.json()['threads']:
            if not bare_records['threads'].get(entry['id']):
                bare_records['threads'][entry['id']] = entry
        for entry in r.json()['users']:
            if not bare_records['users'].get(entry['id']):
                bare_records['users'][entry['id']] = entry
        for entry in r.json()['webhooks']:
            if not bare_records['webhooks'].get(entry['id']):
                bare_records['webhooks'][entry['id']] = entry
        generate_log_entries(r.json())
        last_id = int(r.json()['audit_log_entries'][-1]['id'])
        cycles += 1
        
        # Check against start time
        if snowflake_to_timestamp(last_id) < START_TIME:
            print(cycles)
            break
        time.sleep(1)
        # except Exception as e:
        #     print(e)

    with open(f"Output-RAW-{DATE_FILE}.json", "w", encoding='utf-8') as output:
        output.write(json.dumps(bare_records, indent=4, sort_keys=True))
        output.close()

    print("Logging Finished!")


main()
