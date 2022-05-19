#date: 2022-05-19T17:19:08Z
#url: https://api.github.com/gists/32a69ea61eaa839ff18b2adf77b7283e
#owner: https://api.github.com/users/Soyuzbek

from typing import Optional

import pydantic


class AllOptional(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespaces, **kwargs):
        annotations = namespaces.get('__annotations__', {})
        for base in bases:
            annotations.update(base.__annotations__)
        for field in annotations:
            if not field.startswith('__'):
                annotations[field] = Optional[annotations[field]]
        namespaces['__annotations__'] = annotations
        return super().__new__(mcs, name, bases, namespaces, **kwargs)


def partial(cls):
    """
    Class factory that generates pydantic model for partial update requests.
    usage: partial(<pydantic Model>)
    explanation:
        consider:
        class ItemUpdate(BaseModel):
            name: str
            count: int
    then:
    ItemPartialUpdate = partial(ItemUpdate)
    is equivalent to:
    class ItemPartialUpdate(BaseModel):
        name: Optional[str]
        count: Optional[int]

    Note: the name of newly generated model will derive
    from the model which was given as an argument.
    """
    name = cls.__name__.replace('Update', 'PartialUpdate')
    partial_class = AllOptional(name, cls, {})
    return partial_class
