#date: 2022-01-28T17:00:40Z
#url: https://api.github.com/gists/14c219f7de494ccb91a67c1c166e19cc
#owner: https://api.github.com/users/vkhangpham

import re

def process(s):
    res = re.sub(r'(\w)(\()(\w)', '\g<1> \g<2>\g<3>', s)
    res = re.sub(r'(\w)([),.:;]+)(\w)', '\g<1>\g<2> \g<3>', res)
    res = re.sub(r'(\w)(\.\()(\w)', '\g<1>. (\g<3>', res)
    res = re.sub(r'\s+', ' ', res)
    res = res.strip()
    return res

def add_token_column(example):
    return {
        'raw_address': [item.strip() for item in example['raw_address']],
        'tokens': [process(item).split() for item in example['raw_address']]
        }

def clean(example):
    return {
        'POI': [process(item.split('/')[0]).split() for item in example['POI/street']],
        'STR': [process(item.split('/')[1]).split() for item in example['POI/street']],
        'labels': [['O']*len(item) for item in example['tokens']]
        }

raw_train = raw_train.map(add_token_column, batched=True)
raw_train = raw_train.map(clean, batched=True)

raw_test = raw_test.map(add_token_column, batched=True)
raw_test = raw_test.map(clean, batched=True)