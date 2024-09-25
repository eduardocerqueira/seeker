#date: 2024-09-25T16:46:12Z
#url: https://api.github.com/gists/f5f4463efbbdfd8f40b68f399c579f08
#owner: https://api.github.com/users/lovemycodesnippets

import jsonschema
from jsonschema import validate

def validate_profile(profile, schema):
    try:
        validate(instance=profile, schema=schema)
        print("Profile is valid.")
    except jsonschema.exceptions.ValidationError as err:
        print("Profile is invalid:", err)

# Example usage
profile = generate_profile_with_function_calling()
validate_profile(json.loads(profile), schema)