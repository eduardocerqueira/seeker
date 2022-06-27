#date: 2022-06-27T17:03:15Z
#url: https://api.github.com/gists/532bca5509c810317febffe6354400c8
#owner: https://api.github.com/users/johnny-godoy

from __future__ import annotations

import tabulate


def tabulate_query(query: Query, **kwargs) -> str:
    return tabulate.tabulate(query, headers=(column.get('name') for column in query.column_descriptions), **kwargs)
