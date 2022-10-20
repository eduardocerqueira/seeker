#date: 2022-10-20T17:33:19Z
#url: https://api.github.com/gists/d11723b13dd2e7c373e9a5d0200fdb39
#owner: https://api.github.com/users/dutc

from dataclasses import dataclass

@dataclass
class Ctx:
    mode : bool = True

    def __post_init__(self):
        if (mode := self.mode):
            cls = type(self)
            self.__class__ = type(
                f'fake_{cls.__name__}',
                cls.__mro__,
                {
                    **cls.__dict__,
                    '__exit__': lambda *_: print(f'{mode = }'),
                }
            )

    __enter__ = lambda s: s
    __exit__  = lambda s, *_: None

with Ctx(mode=False) as obj:
    print(f'{obj = }')

with Ctx(mode=True) as obj:
    print(f'{obj = }')
