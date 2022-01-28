#date: 2022-01-28T17:08:32Z
#url: https://api.github.com/gists/94e11049e5fdbd3ffe0b9d8336de3e87
#owner: https://api.github.com/users/akshay-ravindran-96

import re

re.sub(r'http\S+', '', text) # URL
text = text.lower() # Lower Case
re.sub(r'[^\w\s]', '', text) # Punctuations
re.sub( r'[0-9]', '', text) # Digits