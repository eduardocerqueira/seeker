#date: 2022-10-19T17:29:04Z
#url: https://api.github.com/gists/e3e7a2321db1f1345ed03d3f22a891a7
#owner: https://api.github.com/users/Ripeey

def index_list(_list, key, skip_dup = True):
    """
    Converts a list into a dict and adds index based on a given key

    :param _list: List to convert to indexed dictionary.
    :param key: Key needs to be str|int|list[str|int]|callable. 
            list[str|int] :
                Create child nodes in order with keys within list. Note, provided list of keys has to be 
                within same level of dict(item). 
                Eg: [{'a' : 'key1', 'b': 'key2'}, ...] -> {'key1' : {'key2' : {...} }}}
            Callable :
                If callable then a key can be a set as well.
                A `set` will be used for {'key0' : [item], 'key1': [item]} and set(key0,key1).
                If a set is provided then the same item will be sent to "set" of keys holding list instead.

    :return dictionary:
    """
    dictionary = dict()

    if not isinstance(_list, list):
        raise TypeError()

    for item in _list:
        if callable(key):
            _key = key(item)
            if isinstance(_key, set):
                for _ in _key: 
                    if _ in dictionary:
                        dictionary[_].append(item)
                    else:
                        dictionary[_] = [item]
            elif isinstance(_key, list):
                _key.reverse()
                chunk = deepcopy(item)
                for _ in _key:
                    chunk = {item[_] : chunk}
                dictionary.update(chunk)
            else:
                if not skip_dup and _key in dictionary:
                    raise ValueError(f'Duplicate value provided as key.')
                dictionary[_key] = item
        elif isinstance(key, list):
            key.reverse()
            chunk = deepcopy(item)
            for _ in key:
                chunk = {item[_] : chunk}
            dictionary.update(chunk)
        else:
            if not item.get(key):
                raise KeyError(f'Key `{key}` not found within index list[dict]')
            if not skip_dup and key in dictionary:
                raise ValueError(f'Duplicate item found as key within list[dict].')
            
            dictionary[item.get(key)] = item

    return dictionary
