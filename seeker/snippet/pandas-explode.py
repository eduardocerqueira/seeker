#date: 2021-12-06T16:49:45Z
#url: https://api.github.com/gists/d185b619a73c928556276fde5a18522a
#owner: https://api.github.com/users/Shawe82

import pandas as pd
n_rows = 3
result = pd.DataFrame(
    {"a": [list(range(1 + i ** 2)) for i in range(n_rows)], "b": list(range(n_rows))}
).explode("a").astype({'a':int})