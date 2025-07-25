#date: 2025-07-25T17:09:59Z
#url: https://api.github.com/gists/1217e3d01a83e4353d3bccacb12d588b
#owner: https://api.github.com/users/MichaelGift

class TypeCommand(EditorCommand):
    def __init__(self, editor, char_to_type):
        self._editor = editor # The Receiver
        self._char = char_to_type
        self._prev_state = "" # To store editor content before action

    def execute(self):
        self._prev_state = self._editor.get_content() # Save current state for undo
        self._editor.type_char(self._char)
        print(f"  Executed: Typed '{self._char}'. Editor: '{self._editor.get_content()}'")

    def undo(self):
        self._editor.set_content(self._prev_state) # Restore previous state
        print(f"  Undone: Typed '{self._char}'. Editor: '{self._editor.get_content()}'")

# Similar classes would exist for DeleteCommand, PasteCommand, etc.