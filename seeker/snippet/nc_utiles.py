#date: 2022-12-20T17:02:51Z
#url: https://api.github.com/gists/3bc14e1de61fbe439e3e06d6b3e25ccf
#owner: https://api.github.com/users/bzah

# License: This code is licensed under the terms of the APACHE 2 License (https://www.apache.org/licenses/LICENSE-2.0.html)
# Copyright (C) 2022 Aoun Abel aoun.abel@gmail.com

def get_groups(ds: nc.Dataset) -> list[str]:
    """
    Get all groups within a netcdf Dataset.

    Return a list of all the groups.
    """

    def _get_groups(group, acc) -> list[str]:
        if group.groups:
            for inner_group in group.groups:
                _get_groups(group[inner_group], acc)
            return acc
        else:
            acc += [group.path]
        return acc

    return _get_groups(ds, [])


def get_paths(ds: nc.Dataset) -> list[str]:
    """
    Get all paths within a netcdf Dataset.

    Return a list of all the groups.
    """
    groups = get_groups(ds)
    acc = []
    for gr in groups:
        for group_var in ds[gr].variables:
            acc += [f"{gr}/{group_var}"]
    return acc


def whereis(ds: nc.Dataset, query: str) -> dict[str, str]:
    """
    Try to find `query` in a netcdf. Search paths and attributes

    Return a list of all paths (group/variable) where the query was found.
    
    Usage
    >>> whereis(query= "radiance", data=data)
    """
    acc = []
    query = query.upper()
    for att in ds.ncattrs():
        if query in att.upper() or query in str(ds.getncattr(att)).upper():
            acc += [{"path": "/", "attrs": att}]
    for group_path in get_paths(ds):
        if query in group_path.upper():
            acc += [{"path": group_path}]
        else:
            for att in ds[group_path].ncattrs():
                if (
                    query in att.upper()
                    or query in str(ds[group_path].getncattr(att)).upper()
                ):
                    acc += [{"path": group_path, "attrs": att}]
    return acc
