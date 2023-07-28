#date: 2023-07-28T16:45:00Z
#url: https://api.github.com/gists/5a6569f10500aafa2d68ff2848f9306b
#owner: https://api.github.com/users/shawn42

from functools import wraps
from typing import Any, Callable, TypeVar, Type, cast, get_type_hints

T = TypeVar('T')


class AppContext:
    def __init__(self) -> None:
        self.dependencies: dict[type, Callable[..., Any]] = {}

    def register(self, dependency: Any) -> None:
        if type(dependency) == type:
            self.dependencies[dependency] = dependency
        else:
            self.dependencies[dependency.__annotations__['return']] = dependency

    def resolve(self, dependency_type: Type[T]) -> T:
        if dependency_type not in self.dependencies:
            raise ValueError(f"Dependency not registered for type: {dependency_type}")
        return cast(T, self.dependencies[dependency_type]())

    def __getitem__(self, key: Type[T]) -> T:
        return self.resolve(key)

    def __setitem__(self, reg: Callable[..., Type[T]]) -> None:
        return self.register(reg)


context = AppContext()


def inject_dependencies_from(context: AppContext) -> Callable[[Type[T]], Type[T]]:
    def decorator(cls: Type[T]) -> Type[T]:
        constructor_hints = get_type_hints(cls.__init__)
        original_init = cls.__init__
        @wraps(cls)
        def init_with_dependencies(self, *args: Any, **kwargs: Any) -> None: #type: ignore
            resolved_args = dict((k, context.get(v_type)) for (k, v_type) in constructor_hints.items())
            original_init(self, **resolved_args, **kwargs)

        cls.__init__ = init_with_dependencies # type: ignore
        return cls

    return decorator

class Settings:
    @property
    def db_name(self) -> str:
        return "example_db"

# Example usage
@inject_dependencies_from(context=context)
class Database:
    def __init__(self, settings: Settings):
        self.db_name = settings.db_name

def use_db(db: Database) -> None:
    print(db.db_name)  # Output: "example_db"

if __name__ == "__main__":
    # Register dependencies
    # def create_settings() -> Settings:
    #     return Settings()
    # uses the return type of the function
    # context.register(create_settings)

    # or just
    context.register(Settings)
    context.register(Database)

    # Resolve dependencies
    db_instance = context[Database]
    use_db(db_instance)  # this passes mypy Type checking