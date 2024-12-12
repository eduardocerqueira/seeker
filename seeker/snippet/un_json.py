#date: 2024-12-12T16:53:06Z
#url: https://api.github.com/gists/5845eca57fb0207b0c723c63ad53122f
#owner: https://api.github.com/users/chapmanjacobd

def un_json(input_dict: dict[str, str | float]):
    processed_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, str) and value.startswith(('{', '[')):
            try:
                processed_dict[key] = json.loads(value)
            except json.JSONDecodeError:
                processed_dict[key] = value
        else:
            processed_dict[key] = value
    return processed_dict
