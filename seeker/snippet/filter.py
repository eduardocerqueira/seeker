#date: 2024-02-01T16:52:31Z
#url: https://api.github.com/gists/7325d1ebddbfea0cff86519111dc5e74
#owner: https://api.github.com/users/zackbunch

def filter_list_of_dicts_decorator(key_to_filter=None, filter_pattern=None):
    def decorator(func):
        def wrapper(data_list, *args, **kwargs):
            if key_to_filter is None:
                return func(data_list, *args, **kwargs)

            filtered_list = []
            for item in data_list:
                if key_to_filter in item and isinstance(item[key_to_filter], str):
                    if filter_pattern is None or not item[key_to_filter].startswith(filter_pattern):
                        filtered_list.append(item)

            return func(filtered_list, *args, **kwargs)

        return wrapper
    return decorator

# Usage of the decorator
@filter_list_of_dicts_decorator(key_to_filter="name", filter_pattern="nx-")
def process_data(data_list):
    # Your data processing logic here
    return data_list
