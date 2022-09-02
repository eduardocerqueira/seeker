#date: 2022-09-02T17:17:09Z
#url: https://api.github.com/gists/8934c9dcb62e5b91c50ba6654b68474e
#owner: https://api.github.com/users/yousafmufc

import pandas as pd
wind_df = pd.read_csv(data_filename,engine='python',encoding='latin1') #adding the extra parameters because of encoding issues
wind_df