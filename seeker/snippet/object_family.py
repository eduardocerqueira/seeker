#date: 2025-07-21T17:04:02Z
#url: https://api.github.com/gists/3ecbb1651171201a67a6946c30d7a99d
#owner: https://api.github.com/users/OliverStrait

"""Family is a tree structure that retend reference of an object"""

from __future__ import annotations

from typing import Callable, Self, TypeAlias, Any, Protocol, Any
from abc import ABC, abstractmethod
import weakref

from ..modules.reference import WeakRefDesc


VALIDATOR: TypeAlias = Callable[["FamilyTree", "FamilyTree"], bool]


class FamilyKin(Protocol):
    """Type of class having memory adress for family node"""
    nufamily: FamilyTree


class RootError(TypeError): ...

class NodeError(SyntaxError): ...

class ValidatorError(SyntaxError): ...

class DeadNodeError(SyntaxError): ...

class CopyError(NodeError): ...

class FamilyTree:

    __owner: Any = WeakRefDesc()
    __root: Self
    __parent: Self | None
    __children: list[Self]
    __validator: Callable

    def __init__(self, owner: Any, validator: VALIDATOR|None = None):
        self.__root = self
        self.__owner = owner
        weakref.finalize(owner, self.__close)

        self.__parent = None
        self.__children = []
        self.validator = validator if validator else lambda a, b: True 

    @property
    def validator(self):
        """Function that can validity compability of child tree.
        Validator should be function that is a state agnostic
        like type test.
        """
        return self.__validator
    
    @validator.setter
    def validator(self, val: VALIDATOR):

        if isinstance(val, Callable):

            ret = val(self.__owner, self.__owner)
            if isinstance(ret, bool):
                self.__validator = val
            else:
                raise ValidatorError(f"Validator does not retun bool values: [{ret}]")
        else:
            raise ValueError("Validator is not a callable.")

    def validate_new_node(self, node: Self):
        """Valid node
        - is alive (has reference to owner)
        - is it's own root `.__root = self`
        - is valid by validator function
        """
        if node.__root is node:
            if self.__validator(self.__owner, node.__owner):
                return True
            else:
                raise ValidatorError("External validation failed:", self.__owner, node.__owner)
        else:
            a, b = self.__owner, node.__owner
            if a is None or b is None:
                raise DeadNodeError(f"Try to use dead node. [self: {a}, tested: {b}]")
            else:
                root = node.__root
                raise NodeError(f"@{node} is part of [{root}] tree. Nodes can be part of only one tree"+
                            f"owners: [new: {node.__owner}, root: {root.__owner}]")

    def add_child(self, child:Self):
        self.validate_new_node(child)
        self.__dirty = True
        self.__children.append(child)
        child.__root = self.__root
        child.__parent = self
        child.__downstream_new_root(self.__root)

    def detach(self):
        """Detache all connections to up/downstream"""
        self.detach_children()
        self.detach_from_upstream()

    def detach_children(self):
        """Detach all children nodes and update downstream"""
        for child in self.__children:
            child.detach_from_upstream()
            child.__downstream_new_root(child)

    def detach_from_upstream(self):
        """Detach from parent tree and make self as root"""
        if self.__parent is not None:
            self.dirty_upstream()
            self.__root = self
            self.__root.__dirty = True
            self.__parent.__children.remove(self)
        self.__parent = None

    def dirty_upstream(self):
        self.__dirty = True
        if not self.__root is self:
            return self.__parent.dirty_upstream()
        
    def __close(self):
        """Destrroying method called when owner is deleted"""
        self.detach()
        self.__children.clear()
        self.__root = None

    def __downstream_new_root(self, new_root:Self):
        """Inform new root towards leafs"""
        for  child in self.__children:
            child.__root = new_root
            child.__downstream_new_root(new_root)
        

    def __str__(self):
        return f" {self.__class__.__name__} serving {self.__owner}"

    def __deepcopy__(self, memo: dict):
        """Create a new instance of Tree structure detached from old tree structure"""
        if len(memo) == 0:
            raise CopyError(f"@[{self}] Creation of standalone copies is prohibited")
        owner =  memo[id(self.__owner)]
        return self.__class__(owner, self.__validator)

    def __copy__(self):
        """No copies is allowed because those can replace nodes and cause looping trees and recursion errors"""
        raise CopyError(f"@[{self}] Creation of standalone copies is prohibited")
    
    @property
    def family(self):
        return [child.__owner for child in self.__children]
    
    @property
    def root(self):
        """Return root of node.
        :Warning:
        - Avoid saving nodes in longterm because nodes will lost validity when owner is deleted.
        """
        return self.__root
    @property
    def domain(self):
        """Return whole downstream-domain. all subfamilies"""
        if self.__root is self:
            if self.__dirty:
                self.__domain_cache = self.family
            return self.__domain_cache
        else:
            return self.family