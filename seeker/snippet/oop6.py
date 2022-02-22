#date: 2022-02-22T16:56:17Z
#url: https://api.github.com/gists/eae3accf0db1c471efc4b54eca531e0b
#owner: https://api.github.com/users/jimmy-law

import pandas as pd
sdb = SportsDB()
last_5_games = sdb.get_last_5_games_for_league("English Premier League")
pd.DataFrame(last_5_games)