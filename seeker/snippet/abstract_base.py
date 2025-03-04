#date: 2025-03-04T17:03:43Z
#url: https://api.github.com/gists/337dea07586ed0ecb5fe5d89cb629ebf
#owner: https://api.github.com/users/nottrobin

class AbstractBase(ABC):
    """
    An abstract base class that refuses to be instantiated directly. Because, even
    though "ABC" is in theory an "abstract base class", it really doesn't fit the
    definition of "abstract" if it doesn't block direct instantiation.
    (Based on Aran Fey's answer: https://stackoverflow.com/a/50100282/613540)
    """

    abstract: bool = True

    def __init_subclass__(cls, /, abstract=False, **kwargs):
        """
        Direct children of AbstractBase are implicitly abstract (of type "ABCMeta"),
        so they should inherit the metaclass "ABCMeta" from AbstractBase.

        More distant descentents, however, should be "real" classes of type "type"
        unless they are explicitly designated as metaclasses
        with `metaclass=ABCMeta`.
        """

        parent = cls.mro()[1]

        if parent is not AbstractBase:
            cls.abstract = abstract

        return super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        if cls.abstract:
            raise TypeError(
                f"AbstractBase meta-classes may not be directly instantiated."
            )

        return super().__new__(cls, *args, **kwargs)