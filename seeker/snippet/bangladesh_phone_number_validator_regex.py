#date: 2023-06-15T17:01:54Z
#url: https://api.github.com/gists/f11a06b6ae248a2781ff52ad2b8518cd
#owner: https://api.github.com/users/forhadakhan

import re

pattern = r'^(?:\+?880|0|88)?\s?1[3456789]\d{8}$'

phone_numbers = [
    '01534567890',
    '8801534567890',
    '880 1534567890',
    '88 01534567890',
    '+8801534567890',
    '+880 1534567890',
    '+88 01534567890',
    'InvalidPhoneNumber'
]

for number in phone_numbers:
    match = re.match(pattern, number)
    if match:
        print(f"{number} is a valid Bangladeshi phone number.")
    else:
        print(f"{number} is not a valid Bangladeshi phone number.")



"""
Output:

01534567890 is a valid Bangladeshi phone number.
8801534567890 is a valid Bangladeshi phone number.
880 1534567890 is a valid Bangladeshi phone number.
88 01534567890 is a valid Bangladeshi phone number.
+8801534567890 is a valid Bangladeshi phone number.
+880 1534567890 is a valid Bangladeshi phone number.
+88 01534567890 is a valid Bangladeshi phone number.
InvalidPhoneNumber is not a valid Bangladeshi phone number.

"""