#date: 2023-05-25T16:45:29Z
#url: https://api.github.com/gists/b9daf470d090b4460b13e600ab41d237
#owner: https://api.github.com/users/aviiciii

import re

inpt=input('ID:')
# inpt = '21F3000426'

# writting regex
if re.search(r'^(21|22)(F|DP|DS)([1-3]{1})([0-9]{6})$', inpt):
    print('Valid ID')
else:
    print('Invalid ID')