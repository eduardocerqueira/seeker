#date: 2021-10-08T17:14:06Z
#url: https://api.github.com/gists/ba2d832040b641612e5cacfcd8404f9f
#owner: https://api.github.com/users/peppercat10

import mailbox
import sys
import re

def getEmailAddress(value):
    from_line = re.search(r'(\nFrom:.*\n)', value)
    if not from_line:
        return ""

    result = re.search(r'([\w.+-]+@[\w-]+\.[\w.-]+)', from_line.group(0))
    if not result:
        return ""

    return result.group(0)

def prettyPrintContainer(container_of_addresses : dict):
    for key,value in container_of_addresses.items():
        print(f"{key} | {value} occurrences")
    print()
    print(f"Total e-mails from count above: { sum(container_of_addresses.values()) }")

def main():
    mbox_contents = mailbox.mbox(sys.argv[1])
    values = mbox_contents.values()
    container_of_addresses = {}

    for value in values:
        resulting_email = getEmailAddress(value.as_string())
        if resulting_email in container_of_addresses:
            container_of_addresses[resulting_email] += 1
        else:
            container_of_addresses[resulting_email] = 1
    print(f"Number of e-mails: {len(values)}")
    prettyPrintContainer(container_of_addresses)

main()