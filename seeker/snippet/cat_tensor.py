#date: 2024-12-19T16:56:50Z
#url: https://api.github.com/gists/06c23a25e37e69f3de05c9d031e1512f
#owner: https://api.github.com/users/vmiheer

#!/usr/bin/env python3

import numpy as np
import sys

if len(sys.argv) == 3 and sys.argv[2] == "-f":
    a = np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.float32)
else:
    a = (
        np.frombuffer(open(sys.argv[1], "rb").read(), dtype=np.int64)
        .reshape(-1, 2)
        .transpose(1, 0)
    )
print(a)
