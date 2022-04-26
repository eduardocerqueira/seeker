#date: 2022-04-26T17:02:39Z
#url: https://api.github.com/gists/ed8d6baf384bafba54b809b02773c122
#owner: https://api.github.com/users/anddam

def get_flags(key, sep=" "):
    return list({stripped for item in get_config_var(key).split(sep) if (stripped := item.strip())})