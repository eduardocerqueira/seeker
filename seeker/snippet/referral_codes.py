#date: 2023-02-01T17:10:26Z
#url: https://api.github.com/gists/3cdc13479fcbb0a81d5cb93ee12674b5
#owner: https://api.github.com/users/jeremytregunna

import csv

def generate_referral_code(first_name, last_name):
    first = first_name.lower().encode('ascii', errors='ignore').decode().replace(' ', '-')
    last = last_name.lower().encode('ascii', errors='ignore').decode().replace(' ', '-')

    return f"{first}-{last}".encode('ascii', errors='ignore').decode().replace('(', '').replace(')', '').replace('--', '-')

data = []
with open('file.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # Add "Referral Code" column to each row

    for row in reader:
        row['Referral Code'] = generate_referral_code(row['Contact ID'], row['First Name'], row['Last Name'])
        data.append(row)
        print(row['Referral Code'])

with open('new_file.csv', newline='', encoding='utf-8', mode='w') as csvfile:
    fieldnames = data[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Write each row to new file

    for row in data:
        writer.writerow(row)
