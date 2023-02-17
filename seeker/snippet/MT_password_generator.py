#date: 2023-02-17T16:48:08Z
#url: https://api.github.com/gists/5b9f980ac3a7a059a25622c4d5997e4c
#owner: https://api.github.com/users/Code-Victor

import random

LOWERCASE_ALPHABETS = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE_ALPHABETS = LOWERCASE_ALPHABETS.upper()
DIGITS = "0123456789"
SPECIAL_CHARACTERS = "!@#$%^&*()_+"
PASSWORD_LENGTH = "**********"
lowercase_count = 3
uppercase_count = 3
digits_count = 3
special_count = 3

assert (
 "**********"l "**********"o "**********"w "**********"e "**********"r "**********"c "**********"a "**********"s "**********"e "**********"_ "**********"c "**********"o "**********"u "**********"n "**********"t "**********"  "**********"+ "**********"  "**********"u "**********"p "**********"p "**********"e "**********"r "**********"c "**********"a "**********"s "**********"e "**********"_ "**********"c "**********"o "**********"u "**********"n "**********"t "**********"  "**********"+ "**********"  "**********"d "**********"i "**********"g "**********"i "**********"t "**********"s "**********"_ "**********"c "**********"o "**********"u "**********"n "**********"t "**********"  "**********"+ "**********"  "**********"s "**********"p "**********"e "**********"c "**********"i "**********"a "**********"l "**********"_ "**********"c "**********"o "**********"u "**********"n "**********"t "**********"  "**********"= "**********"= "**********"  "**********"P "**********"A "**********"S "**********"S "**********"W "**********"O "**********"R "**********"D "**********"_ "**********"L "**********"E "**********"N "**********"G "**********"T "**********"H "**********"
)

crude_password = "**********"
"".join(random.sample(LOWERCASE_ALPHABETS, lowercase_count))
+ "".join(random.sample(UPPERCASE_ALPHABETS, uppercase_count))
+ "".join(random.sample(DIGITS, digits_count))
+ "".join(random.sample(SPECIAL_CHARACTERS, special_count))
)
password = "**********"
print(password)