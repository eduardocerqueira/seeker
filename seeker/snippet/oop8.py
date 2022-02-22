#date: 2022-02-22T17:01:08Z
#url: https://api.github.com/gists/8179005994df89936b7b30949d0601de
#owner: https://api.github.com/users/jimmy-law

    # constructor
    def __init__(self, timezone="Europe/London"):
        self.display_timezone = pytz.timezone(timezone)