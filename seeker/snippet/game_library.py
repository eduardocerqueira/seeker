#date: 2026-01-28T17:09:31Z
#url: https://api.github.com/gists/fd29cfec003f330de11b2c6b6ff9abfc
#owner: https://api.github.com/users/Jacob-Havlin

import pandas as pd

game_collection = {
    'game_id': [1, 2, 3, 4, 5],
    'title': ['The Legend of Zelda', 'Cyberpunk 2077', 'Minecraft', 'Hades', 'Stardew Valley'],
    'player_count': [1, 1, 10, 1, 4],
    'price': [59.99, 29.99, 26.95, 24.99, 14.99],
    'release_date': ['2017-03-03', '2020-12-10', '2011-11-18', '2020-09-17', '2016-02-26']
}

df = pd.DataFrame(game_collection)

df.to_csv('game_library.csv', index=False)
