#date: 2022-07-14T17:08:17Z
#url: https://api.github.com/gists/6288c2c2fc1e9c538ccce0702a2637ea
#owner: https://api.github.com/users/alirezaimn

from abc import ABC, abstractmethod

class A(ABC):
    @classmethod
    def __subclasshook__(cls,subclass):
        if cls is A:
            subclass_dict = subclass.__mro__[0].__dict__
            cls_dict = cls.__mro__[0].__dict__

            for method_name, method in cls_dict.items():
                if hasattr(method,'__annotations__'):
                    if method_name not in subclass_dict or \
                    subclass_dict[method_name].__annotations__ != cls_dict[method_name].__annotations__:
                            return False
            return True
    
    @abstractmethod
    def a(self, str_param: str) -> list:
        raise NotImplementedError

class B(A):
    def a(self, str_param: str) -> list:
        pass

class C(A):
    def a(self, str_param: int) -> list:
        pass

print(issubclass(B,A)) #Prints True
print(issubclass(C,A)) #Prints False