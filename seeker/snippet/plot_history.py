#date: 2022-04-15T17:07:40Z
#url: https://api.github.com/gists/ddbf634a7bb810631c2ca6507825e80a
#owner: https://api.github.com/users/algonacci

import pandas as pd

evaluation = pd.DataFrame(model.history.history)
evaluation[['accuracy', 'val_accuracy']].plot()
evaluation[['loss', 'val_loss']].plot()