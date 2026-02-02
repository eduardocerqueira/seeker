#date: 2026-02-02T17:17:27Z
#url: https://api.github.com/gists/03ab2e62c0da7cd38efcbe9cd47f828b
#owner: https://api.github.com/users/RareBird15

"""
Reads compromised credentials, extracts usernames,
writes them to text and JSON files, and replaces the
original password file with a signature.
"""

# Import required modules
import csv
import json

# Create list for storing usernames of compromised users
compromised_users = []

# Get usernames of compromised users and store them for later work
 "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********". "**********"c "**********"s "**********"v "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"f "**********"i "**********"l "**********"e "**********": "**********"
  password_csv = "**********"
  for row in password_csv: "**********"
    password_row = "**********"
    compromised_users.append(password_row["Username"])

# Write retrieved usernames to file
with open("compromised_users.txt", "w") as compromised_user_file:
  for user in compromised_users:
    compromised_user_file.write(user + "\n")

# Write message to boss
with open("boss_message.json", "w") as boss_message:
  boss_message_dict = {
    "recipient": "The Boss",
    "message": "Mission Success"
  }
  json.dump(boss_message_dict, boss_message)

# Remove passwords.csv
 "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"n "**********"e "**********"w "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********". "**********"c "**********"s "**********"v "**********"" "**********", "**********"  "**********"" "**********"w "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"n "**********"e "**********"w "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********"_ "**********"o "**********"b "**********"j "**********": "**********"
  slash_null_sig = """\
 _  _     ___   __  ____             
/ )( \   / __) /  \(_  _)            
) \/ (  ( (_ \(  O ) )(              
\____/   \___/ \__/ (__)             
 _  _   __    ___  __ _  ____  ____  
/ )( \ / _\  / __)(  / )(  __)(    \ 
) __ (/    \( (__  )  (  ) _)  ) D ( 
\_)(_/\_/\_/ \___)(__\_)(____)(____/ 
        ____  __     __   ____  _  _ 
 ___   / ___)(  )   / _\ / ___)/ )( \
(___)  \___ \/ (_/\/    \\___ \) __ (
       (____/\____/\_/\_/(____/\_)(_/
 __ _  _  _  __    __                
(  ( \/ )( \(  )  (  )               
/    /) \/ (/ (_/\/ (_/\             
\_)__)\____/\____/\____/
  """
  
  new_passwords_obj.write(slash_null_sig)
  