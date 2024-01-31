#date: 2024-01-31T17:04:51Z
#url: https://api.github.com/gists/2667af84153032f4b8cb579eb7fc43bb
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations
from typing import Any, Type, Literal, overload, TYPE_CHECKING, TypeVar, Optional

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

T = TypeVar("T")

class LoInst:
    pass

class QiPartialT(Protocol):
    # region    qi()

    @overload
    def qi(self, atype: Type[T]) -> Optional[T]:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface

        Returns:
            T | None: instance of interface if supported; Otherwise, None
        """
        ...

    @overload
    def qi(self, atype: Type[T], raise_err: Literal[True]) -> T:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface
            raise_err (bool, optional): If True then raises MissingInterfaceError if result is None. Default False

        Raises:
            MissingInterfaceError: If 'raise_err' is 'True' and result is None

        Returns:
            T: instance of interface.
        """
        ...

    @overload
    def qi(self, atype: Type[T], raise_err: Literal[False]) -> Optional[T]:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface
            raise_err (bool, optional): If True then raises MissingInterfaceError if result is None. Default False

        Raises:
            MissingInterfaceError: If 'raise_err' is 'True' and result is None

        Returns:
            T | None: instance of interface if supported; Otherwise, None
        """
        ...

    # endregion qi()



class QiPartial(QiPartialT):
    def __init__(self, component: Any, lo_inst: LoInst):
        self.__lo_inst = lo_inst
        self.__component = component
    
    @overload
    def qi(self, atype: Type[T]) -> Optional[T]:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface

        Returns:
            T | None: instance of interface if supported; Otherwise, None
        """
        ...

    @overload
    def qi(self, atype: Type[T], raise_err: Literal[True]) -> T:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface
            raise_err (bool, optional): If True then raises MissingInterfaceError if result is None. Default False

        Raises:
            MissingInterfaceError: If 'raise_err' is 'True' and result is None

        Returns:
            T: instance of interface.
        """
        ...

    @overload
    def qi(self, atype: Type[T], raise_err: Literal[False]) -> Optional[T]:  # pylint: disable=invalid-name
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type such as XInterface
            raise_err (bool, optional): If True then raises MissingInterfaceError if result is None. Default False

        Raises:
            MissingInterfaceError: If 'raise_err' is 'True' and result is None

        Returns:
            T | None: instance of interface if supported; Otherwise, None
        """
        ...
    
    def qi(self, atype: Any, raise_err: bool = False) -> Any:
        """
        Generic method that get an interface instance from  an object.

        Args:
            atype (T): Interface type to query obj for. Any Uno class that starts with 'X' such as XInterface
            raise_err (bool, optional): If True then raises MissingInterfaceError if result is None. Default False

        Raises:
            MissingInterfaceError: If 'raise_err' is 'True' and result is None

        Returns:
            T | None: instance of interface if supported; Otherwise, None

        Note:
            When ``raise_err=True`` return value will never be ``None``.
        """
        return atype

clz = QiPartial(None, LoInst())

result = clz.qi(int, True)
reveal_type(result)
