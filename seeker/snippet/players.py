#date: 2021-11-05T17:13:01Z
#url: https://api.github.com/gists/3a32d8e665e266580d05428dba56cf94
#owner: https://api.github.com/users/JakubDotPy

mport logging
from abc import ABC
from abc import abstractmethod
from importlib import import_module

log = logging.getLogger(__name__)


class Player(ABC):

    def __init__(self, name, character):
        self.name = name
        self.character = character

    @classmethod
    def from_file(cls, filename):

        try:
            player_module = import_module(filename)

            player = cls(
                name=getattr(player_module, 'name'),
                character=getattr(player_module, 'character'),
                )
            player.choose_position = getattr(player_module, 'choose_position')

        except FileNotFoundError:
            log.error(f'file {filename} does not exist')
            raise
        except ImportError:
            log.error('error importing module')
            raise
        except TypeError:
            log.error('cant create the player class')
            raise
        else:
            return player

    @abstractmethod
    def choose_position(self, board):
        pass

    @property
    def badge(self):
        return f'{self.__class__.__name__} player {self.name}'

    def __str__(self):
        return self.name


class Human(Player):

    def __init__(self, name, character):
        super().__init__(name, character)

    @staticmethod
    def ask_for_position():
        while True:
            try:
                pos = int(input('Enter the index where you want to play: '))
            except ValueError:
                print('Must be a number!')
            else:
                return pos

    def choose_position(self, board):
        return self.ask_for_position()
