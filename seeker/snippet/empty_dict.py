#date: 2025-11-07T17:13:16Z
#url: https://api.github.com/gists/41154b9027e53abc9104ff949cd9bc80
#owner: https://api.github.com/users/lucasbracher

# Get a dictionary and empty it, leaving just the structure:

def empty_dict(d):
    if isinstance(d, dict):
        return {k: empty_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [empty_dict(d[0])] if d else []
    else:
        return None