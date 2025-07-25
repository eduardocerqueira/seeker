#date: 2025-07-25T17:09:13Z
#url: https://api.github.com/gists/6c9fc46d571b6c93be569b1644a1de6a
#owner: https://api.github.com/users/MichaelGift

from abc import ABC, abstractmethod

class EditorCommand(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self): # Essential for undo/redo features
        pass