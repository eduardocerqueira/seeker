#date: 2024-09-11T17:10:14Z
#url: https://api.github.com/gists/dd08b2806979cb4b7228e7565046f675
#owner: https://api.github.com/users/mypy-play

def handle(err: tuple[type[ValueError] | type[TypeError], ...]):
    try:
        return None
    except err:
        pass


handle((ValueError, TypeError))
