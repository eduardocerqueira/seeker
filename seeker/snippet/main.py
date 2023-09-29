#date: 2023-09-29T16:59:00Z
#url: https://api.github.com/gists/c9da0c6630b53d85642063de8a0e220b
#owner: https://api.github.com/users/mypy-play



class DataEntry(dict):
    pass
    

def get_values(entries: list[dict[str, any]], variable_name: str) -> set[int | str]:
    return set(entry[variable_name] for entry in entries)
    