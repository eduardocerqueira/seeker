#date: 2022-05-12T17:08:04Z
#url: https://api.github.com/gists/29f5d09b76eadd3b760294d2ca22800a
#owner: https://api.github.com/users/jairotunior

from abc import abstractmethod, ABC


class Transformation(ABC):

    def __init__(self, **kwargs):
        assert kwargs.get('name', None), "Se debe definir un nombre."
        assert kwargs.get('suffix', None), "Se debe definir un sufijo"

        self.name = kwargs.get('name')
        self.suffix = "_{}".format(kwargs.get('suffix'))
        self.units_show = kwargs.get('units_show', None)

    @abstractmethod
    def transform(self):
        pass