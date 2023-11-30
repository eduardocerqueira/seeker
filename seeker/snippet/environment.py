#date: 2023-11-30T17:10:09Z
#url: https://api.github.com/gists/0708dc8e8a2493fd13bb5e57d71493fc
#owner: https://api.github.com/users/AlyceOsbourne

from types import MappingProxyType
import dotenv
import os
import sys


def rebind_module_with_class(cls = None, **kwargs):
    if not cls:
        return lambda cls: rebind_module_with_class(cls, **kwargs)
    o = sys.modules[cls.__module__] = cls(**kwargs)
    return o


class Environment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __getattribute__(self, item):
        if item == '__dict__':
            d = super().__getattribute__('__dict__')
            return MappingProxyType({k: os.environ.get(k, v) for k, v in d.items()}) if d else d
        return super().__getattribute__(item)
    
    def __setattr__(self, key, value):
        raise AttributeError('Environment is immutable')

    def __delattr__(self, item):
        raise AttributeError('Environment is immutable')
    
    def __repr__(self):
        return f'{self.__class__.__name__}({" ".join(f"{k}={v!r}" for k, v in self.__dict__.items())})'
    
    def __init_subclass__(cls, bind=False, dotenv_path='.env', **kwargs):
        super().__init_subclass__(**kwargs)
        def __init__(self, **kwargs):
            _kwargs = {k: getattr(self, k) for k in getattr(self, '__annotations__', {})}
            _kwargs.update(kwargs)
            super(self.__class__, self).__init__(**_kwargs)
        cls.__init__ = __init__
        cls.__final__ = True
        if bind:
            o = rebind_module_with_class(cls)
            if dotenv_path:
                dotenv.load_dotenv(dotenv_path)

