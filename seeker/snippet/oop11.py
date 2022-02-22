#date: 2022-02-22T17:05:43Z
#url: https://api.github.com/gists/c561a3686322b9d13a8ac28b2e2963a4
#owner: https://api.github.com/users/jimmy-law

import pandas as pd
sdb = SportsDB(timezone="US/Eastern")
last_5_games = sdb.get_last_5_games_for_league("English Premier League")
pd.DataFrame(last_5_games)