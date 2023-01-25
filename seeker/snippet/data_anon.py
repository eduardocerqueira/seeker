#date: 2023-01-25T17:07:14Z
#url: https://api.github.com/gists/7708b5efa551db7d515fda236e5cbd05
#owner: https://api.github.com/users/ulasozguler

def data_anon(data, whitelist_vals=None, whitelist_keys=None):
    def iter_data(data, key=None):
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = iter_data(v, k)
        elif isinstance(data, list):
            for i, el in enumerate(data):
                data[i] = iter_data(el, key)
        elif isinstance(data, str):
            if (
                data
                and data not in whitelist_vals
                and not data.replace(".", "", 1).isnumeric()
                and key not in whitelist_keys
            ):
                data = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return data

    return iter_data(data)
