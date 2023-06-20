#date: 2023-06-20T17:08:01Z
#url: https://api.github.com/gists/e5d76212cfc174c4299bd428f7b5107e
#owner: https://api.github.com/users/mewh

import json

def json_to_vcard(json_file):
    with open(json_file) as f:
        data = json.load(f)

    vcard = ""
    for item in data["contacts"]["list"]:
        vcard += "BEGIN:VCARD\n"
        vcard += "VERSION:3.0\n"
        vcard += f"N:{item['last_name']};{item['first_name']};;;\n"
        vcard += f"FN:{item['first_name']} {item['last_name']}\n"
        vcard += f"TEL;TYPE=CELL:{item['phone_number']}\n"
        vcard += f"REV:{item['date']}\n"
        vcard += f"X-UNIXTIME:{item['date_unixtime']}\n"
        vcard += "END:VCARD\n"

    return vcard

json_file = "result.json"
vcard_data = json_to_vcard(json_file)
print(vcard_data)
with open('contacts.vcf', 'w') as file:
    file.write(vcard_data)
