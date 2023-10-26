#date: 2023-10-26T16:45:13Z
#url: https://api.github.com/gists/259783551df5139b05b3bee4ca9b0f84
#owner: https://api.github.com/users/Theremsoe

# Enter your code here. Read input from STDIN. Print output to STDOUT

from sys import stdin


def process_raw_contact():
    book = []
    n_contacts = int(line)
    n_stored = 0
    
    # Logic created to read contact info
    for raw_contact in stdin:
        raw_contact = raw_contact.rstrip()
        
        if not raw_contact:
            continue
        
        [name, phone] = raw_contact.split(' ')
        
        book.append((name, phone))
        
        n_stored += 1
        
        if n_contacts == n_stored:
            break
        
    return book


book_address = []

for line in stdin:
    line = line.rstrip()
    
    if not line:
        continue
    
    if line.isdigit():
        book_address = process_raw_contact()
        continue
    
    matches = [f"{name}={phone}" for (name, phone) in book_address if line == name]
    
    if not len(matches):
        print("Not found")
        continue
    
    print(','.join(matches))
    
    
    