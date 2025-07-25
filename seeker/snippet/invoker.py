#date: 2025-07-25T17:10:56Z
#url: https://api.github.com/gists/6df3312bc48366eb96c14ffe1e77d232
#owner: https://api.github.com/users/MichaelGift

class CommandManager:
    def __init__(self):
        self._history = [] # Stores commands for undo
        self._redo_stack = [] # Stores commands for redo

    def execute_command(self, command):
        command.execute()
        self._history.append(command)
        self._redo_stack.clear() # Clear redo stack if a new command is executed

    def undo(self):
        if self._history:
            command = self._history.pop()
            command.undo()
            self._redo_stack.append(command)

    def redo(self):
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.execute() # Re-execute the command
            self._history.append(command)