#date: 2022-06-21T17:21:37Z
#url: https://api.github.com/gists/735a5f76973bbc188e9ccac93517ceb5
#owner: https://api.github.com/users/AAdamczyKK

import random

lower_case_letter0 = chr(random.randint(65, 90))
lower_case_letter1 = chr(random.randint(65, 90))
upper_case_letter0 = chr(random.randint(65, 90))
upper_case_letter1 = chr(random.randint(65, 90))
digit0 = random.randint(0, 9)
digit1 = random.randint(0, 9)

Password = str(lower_case_letter1.lower()) + str(upper_case_letter0) + \
           str(upper_case_letter1)+str(lower_case_letter0.lower()) + str(digit1) + str(digit0)
print(Password)
