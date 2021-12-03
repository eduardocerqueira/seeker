#date: 2021-12-03T17:13:38Z
#url: https://api.github.com/gists/7eac201cbee365540130e5059ca233d1
#owner: https://api.github.com/users/secemp9

import pydoc
import sys

from bpython.pager import page

# Ugly monkeypatching
pydoc.pager = page


class _Helper(object):
    def __init__(self):
        self.helper = pydoc.Helper(sys.stdin, sys.stdout)

    def __repr__(self):
        return ("Type help() for interactive help, "
                "or help(object) for help about object.")

    def __call__(self, *args, **kwargs):
        self.helper(*args, **kwargs)

_help = _Helper()


# vim: sw=4 ts=4 sts=4 ai et
