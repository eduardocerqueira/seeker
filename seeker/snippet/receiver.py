#date: 2025-07-25T17:11:41Z
#url: https://api.github.com/gists/4439e61b3e6298d0c06e356fcf159d1d
#owner: https://api.github.com/users/MichaelGift

class TextEditor:
    def __init__(self):
        self._content = ""

    def type_char(self, char):
        self._content += char

    def delete_char(self):
        if self._content:
            self._content = self._content[:-1] # Remove last character

    def paste_text(self, text):
        self._content += text

    def get_content(self): # Helper for commands to save state
        return self._content

    def set_content(self, content): # Helper for commands to restore state
        self._content = content