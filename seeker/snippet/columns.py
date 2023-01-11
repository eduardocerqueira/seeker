#date: 2023-01-11T17:02:14Z
#url: https://api.github.com/gists/0a1d2fa156ce8780d4a56cfbb0dddf64
#owner: https://api.github.com/users/bschlenk

import os


def columns(data, *, delimiter, width=None, config):
    '''
    config is a list of objects with a `key`, and optionally a `colorize` and `truncate` method
    '''

    # start by extracting out all the columns & getting the width of each

    tc = width if width is not None else os.get_terminal_size().columns
    cols = [[] for _ in range(len(config))]

    for d in data:
        for i, c in enumerate(config):
            k = c['key']
            if type(k) is str:
                l = getattr(d, k)
            else:
                l = k(d)
            cols[i].append(l)

    # get the available space by determining the max size of each non-truncated column
    sizes = [
        max(map(len, cols[i])) if not c.get('truncate', False) else 0
        for i, c in enumerate(config)
    ]

    num_truncated = sum(1 if c.get('truncate', False) else 0 for c in config)
    available_space = (
        (tc - (sum(sizes) + len(delimiter) * (len(config) - 1))) // num_truncated
        if num_truncated > 0
        else 0
    )

    for i, s in enumerate(sizes):
        if s == 0:
            sizes[i] = available_space

    lines = []
    for d in zip(*cols):
        parts = []
        for i, p in enumerate(d):
            c = config[i]
            s = sizes[i]
            l = len(p)
            if c.get('truncate', False):
                p = truncate(p, available_space)
            if c.get('colorize', None):
                p = c['colorize'](p)
            match c.get('justify', 'left'):
                case 'left':
                    m = getattr(p, 'ljust')
                case 'right':
                    m = getattr(p, 'rjust')
                case 'center':
                    m = getattr(p, 'center')
            # account for ansi escape sequences by adding the difference between
            # the colorized and pre-colorized string lengths
            p = m(s + len(p) - l)
            parts.append(p)

        lines.append(delimiter.join(parts))

    return lines