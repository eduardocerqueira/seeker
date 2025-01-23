#date: 2025-01-23T17:11:11Z
#url: https://api.github.com/gists/1afa1a948c18d98e7c51b5826aa913e9
#owner: https://api.github.com/users/btk5301

import json
import yaml

def map_response_to_variables(api_response, config_file_path):
    """
    Maps API response data to variables using a configuration file.

    :param api_response: dict, API response data
    :param config_file_path: str, path to the configuration file (JSON or YAML)
    :return: dict, mapped variables
    """
    # Load the configuration file
    with open(config_file_path, 'r') as file:
        if config_file_path.endswith('.json'):
            config = json.load(file)
        elif config_file_path.endswith('.yaml') or config_file_path.endswith('.yml'):
            config = yaml.safe_load(file)
        else:
            raise ValueError("Unsupported configuration file format. Use JSON or YAML.")

    # Map response to variables
    mapped_variables = {}
    for var_name, response_path in config.items():
        keys = response_path.split('.')
        value = api_response
        try:
            for key in keys:
                if isinstance(value, list):  # If it's a list, convert key to an index
                    key = int(key)
                value = value[key]
            mapped_variables[var_name] = value
        except (KeyError, IndexError, ValueError, TypeError):
            mapped_variables[var_name] = None  # Set to None if path is invalid

    return mapped_variables

# Example Usage
if __name__ == "__main__":
    # Example API response
    api_response = {
        "data": {
            "user": {
                "id": 123,
                "name": "John Doe",
                "address": {
                    "city": "New York",
                    "zip": "10001"
                }
            },
            "settings": {
                "theme": "dark"
            }
        }
    }

    # Example configuration file content (JSON or YAML)
    # config.json or config.yaml:
    # {
    #     "user_id": "data.user.id",
    #     "user_name": "data.user.name",
    #     "user_city": "data.user.address.city",
    #     "user_theme": "data.settings.theme"
    # }

    # Path to the configuration file
    config_file_path = "config.json"

    # Map response to variables
    result = map_response_to_variables(api_response, config_file_path)
    print(result)
