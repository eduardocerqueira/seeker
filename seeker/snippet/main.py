#date: 2022-06-24T16:46:07Z
#url: https://api.github.com/gists/397fff99661e77bac5ff841b2b9075dc
#owner: https://api.github.com/users/pschanely

from dataclasses import dataclass

class HasConsistentHash:
    '''
    A mixin to enforce that classes have hash methods that are consistent
    with thier equality checks.
    '''
    def __eq__(self, other: object) -> bool:
        '''
        post: implies(__return__, hash(self) == hash(other))
        '''
        raise NotImplementedError

@dataclass
class Apples(HasConsistentHash):
    '''
    Uses HasConsistentHash to discover that the __eq__ method is 
    missing a test for the `count` attribute.
    '''
    count: int
    kind: str
    def __hash__(self):
        return self.count + hash(self.kind)
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Apples) and self.kind == other.kind
    def __repr__(self):
        return f'Apples({self.count!r}, {self.kind!r})'
