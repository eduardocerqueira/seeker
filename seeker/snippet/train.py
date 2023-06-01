#date: 2023-06-01T17:07:08Z
#url: https://api.github.com/gists/5e3de8e0300df203ed34ce20ac6ba042
#owner: https://api.github.com/users/almost

import os, re
from toastkenizer import Toastkenizer

MAX_TOKENS = "**********"
INPUT = "inputdata"

def readfromdir(d):
    for file in os.listdir(d):
        yield(file, open(os.path.join(d, file), "rb").read().decode("latin-1"))

tokenizer = "**********"

 "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"l "**********"i "**********"s "**********"t "**********") "**********"  "**********"< "**********"  "**********"M "**********"A "**********"X "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"S "**********": "**********"
    pair_counts = {}
    total_counts = {}
    max_found = None
    for filename, data in readfromdir(INPUT):
        for line in data.split("\n"):
            weight = 100 if "TOAST" in line.upper() else 1
            for word in re.split(r"\b", line):
                # print("".join(tokenizer.normalize(line)))
                tokens = "**********"
                prev_token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"p "**********"r "**********"e "**********"v "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                    continue
                total_counts[prev_token] = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                    total_counts[token] = "**********"
                    # Make a new token from this pair
                    new_token = "**********"
                    # Check if it already exists
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"e "**********"w "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"_ "**********"l "**********"o "**********"o "**********"k "**********"u "**********"p "**********": "**********"
                        pair_counts[new_token] = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"f "**********"o "**********"u "**********"n "**********"d "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********"  "**********"o "**********"r "**********"  "**********"p "**********"a "**********"i "**********"r "**********"_ "**********"c "**********"o "**********"u "**********"n "**********"t "**********"s "**********"[ "**********"n "**********"e "**********"w "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"] "**********"  "**********"> "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"f "**********"o "**********"u "**********"n "**********"d "**********"[ "**********"1 "**********"] "**********": "**********"
                            max_found = "**********"
                    prev_token = "**********"

    assert max_found is not None

    (new_token, count) = "**********"

    tokenizer.add_token(new_token)
    print(f"Found next best token: "**********"

    open("tokens.txt", "w").writelines(x+"\n" for x in tokenizer.tokens_list)
