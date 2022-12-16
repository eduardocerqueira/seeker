#date: 2022-12-16T16:50:29Z
#url: https://api.github.com/gists/ab0a9170f3df931df59d083bef66463c
#owner: https://api.github.com/users/CastixGitHub

import re


def get_rfc4514(_str, attribute = 'CN'):
    if not isinstance(_str, str):
        # assuming cryptography Certificate object
        _str = cert.subject.rfc4514_string()
    return (
        [v for (k, v) in [
            e.split('=') for e in re.split(
                r'(?<!\\),', _str
            )
        ] if k == attribute] or ['']
    )[0]


if __name__ == '__main__':
    _str = 'CN=Sc\,hloß.de,O=Me'
    assert get_rfc4514(_str) == 'Sc\\,hloß.de'
    assert 'Sc\\,hloß.de' == 'Sc\,hloß.de'  # it's the same thing
    assert 'Me' == get_rfc4514(_str, 'O')

