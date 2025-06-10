#date: 2025-06-10T17:06:21Z
#url: https://api.github.com/gists/8790c8e0c2ba374ce5a77bf30f31c5ed
#owner: https://api.github.com/users/nxta-damn

from enum import StrEnum
from typing import Any, Callable, Generator, cast
from inspect import signature, isclass, isgeneratorfunction, isfunction

type Factory = Callable[[Callable[[Any], Any], dict[Any, Any], list[Generator]], Any]
type Dependency = Callable[..., Any] | type[Any]


class BaseScope(StrEnum):
    APP: str = 'app'
    REQUEST: str = 'request'
    SESSION: str = 'session'


def parse_dependency_signature(dependency: Dependency) -> tuple[Any, dict[str, Any]]:
    if isfunction(dependency):
        kw_dependencies, dep_key = {
            k: v.annotation for k, v in signature(dependency).parameters.items()
        }, signature(dependency).return_annotation

    elif isclass(dependency):
        kw_dependencies, dep_key = {
            k: v.annotation for k, v in dependency.__init__.__annotations__.items()
        }, dependency

    return dep_key, kw_dependencies


def make_resolver(
   dependency: Dependency, vars: dict[str, Any], key_type: type
) -> Factory:
    
    def _resolver[T](
        resolve: Callable[[type[Any]], Any], cache: dict[Any, Any], exits: list[Generator]
    ) -> T:
        resolved_vars = {k: resolve(v) for k, v in vars.items()}

        if isgeneratorfunction(dependency):
            gen, solved = dependency(**resolved_vars), gen.send(None)
            exits.append(gen)
        else:
            solved = dependency(**resolved_vars)
            
        cache[key_type] = solved
        return cast('T', solved)

    return _resolver


class Container:
    _registry: dict[Any, tuple[BaseScope, Factory]]
    _cache: dict[Any, Any]
    _exits: list[Generator]

    def __init__(
        self,
        scope: BaseScope,
        parent: 'Container | None' = None,
    ) -> None:
        self._parent = parent
        self._registry = {}
        self._scope = scope
        self._cache = {}
        self._exits = []

    def register(self, dependency: Dependency, scope: BaseScope) -> None: 
        dep_key, dependencies = parse_dependency_signature(dependency)
        resolver = make_resolver(dependency, dependencies, dep_key)
        self._registry[dep_key] = (scope, resolver)

    def resolve[T](self, dependency: type[T]) -> T:
        if dependency in self._cache:
            return self._cache[dependency]

        try:
            dep_scope, resolver = self._registry[dependency]
        except KeyError as _ex:
            raise ValueError(f'{dependency} is not registered in {self._scope} scope') from _ex
        
        if self._scope != dep_scope:
            try:
                return self._parent.resolve(dependency)
            except KeyError as _ex:
                raise ValueError(f'{dependency} is not registered in {self._scope} scope') from _ex
        
        return cast('T', resolver(self.resolve, self._cache, self._exits))