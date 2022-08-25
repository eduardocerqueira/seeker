#date: 2022-08-25T15:37:01Z
#url: https://api.github.com/gists/ab230e71557d0e81e1a432c7e66305f9
#owner: https://api.github.com/users/TheArcherST

import warnings

import os
from shutil import rmtree


class TDException(Exception):
    pass


class AllocationConflict(TDException):
    pass


class TDWarning(Warning):
    pass


class TDSession(str):
    """ Temporary Directory Session """

    _is_closed: bool = False

    def __new__(cls, directory: str):
        self = super().__new__(cls, directory)

        return self

    def open(self):
        try:
            os.mkdir(self)
        except FileExistsError:
            raise AllocationConflict(f"Directory {self} already exists, can't allocate it.")

    def close(self):
        if self._is_closed:
            return
        else:
            rmtree(self)
            self._is_closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return None

    def __del__(self):
        if not self._is_closed:
            warnings.warn(
                TDWarning(f"Session {self} was automatically closed by destructor. To avoid this warning, "
                          f"close session by regular method close, or use session in with expression.")
            )
            self.close()


class TDManager:
    def __init__(self, base_directory: str):
        """ TemporaryDirectoriesManager initialization

        This is object that help you organize temporary files
        in the isolated directories, called TDSession.

        Recommended way to use this object:

        >>> manager = TDManager('./storage')
        >>> with manager() as temp_dir:
        ...     with open(os.path.join(temp_dir, 'test.txt'), 'w'):
        ...         # some manipulations...
        ...         pass

        """

        self._base_directory = base_directory
        self._sessions: list[TDSession] = []
        self._counter = 0

    def _generate_identifier(self) -> str:
        result = self._counter
        self._counter += 1

        result = str(result)

        return result

    @property
    def base_directory(self):
        return self._base_directory

    def new_session(self) -> TDSession:
        return TDSession(os.path.join(self._base_directory, self._generate_identifier()))

    __call__ = new_session
