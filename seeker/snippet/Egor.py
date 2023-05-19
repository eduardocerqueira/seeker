#date: 2023-05-19T16:51:42Z
#url: https://api.github.com/gists/0cbf6111785b52aff7ffbade09ed8f2c
#owner: https://api.github.com/users/daniil-lyakhov

import sys
import json
from collections import defaultdict
import numpy as np

def main():
    path = sys.argv[1]
    with open(path, 'r') as f:
        data = json.load(f)

    aggregated = defaultdict(list)
    for elem in data:
        if 'args' in elem and 'op_name' in elem['args']:
            aggregated[elem['args']['op_name']].append(elem['dur'])
    print('Type_name,len,mean_val')
    for type_name, val in aggregated.items():
        print(type_name, len(val), np.mean(val).astype(np.float16), sep=',\t')

main()
