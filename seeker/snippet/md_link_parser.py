#date: 2025-02-19T16:48:39Z
#url: https://api.github.com/gists/91f4938a39db1b69160f902723e86a9e
#owner: https://api.github.com/users/peter88213

"""Provide a class that converts Markdown links to wikilinks.

Copyright (c) 2025 Peter Triesberger
For further information see https://github.com/peter88213
License: GNU GPLv3 (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

class MdLinkParser:
    """Parser implementing a state machine for Markdown link conversion."""

    def __init__(self):
        self.markup = {
            '[': self.handle_desc_start,
            ']': self.handle_desc_end,
            '(': self.handle_url_start,
            ')': self.handle_url_end,
        }
        self.results = []
        # list of characters and strings
        self.descBuffer = []
        # list of characters, buffering the read-in description
        self.urlBuffer = []
        # list of characters, buffering the read-in URL
        self.state = 'BODY'

    def to_wikilinks(self, text):
        """Return text with Markdown links converted into wikilinks."""
        self.reset()
        self.feed(text)
        self.close()
        return ''.join(self.results)

    def reset(self):
        """Reset the instance. Loses all unprocessed data."""
        self.results.clear()
        self.descBuffer.clear()
        self.urlBuffer.clear()
        self.state = 'BODY'

    def feed(self, data):
        """Feed some text to the parser."""
        for c in data:
            self.markup.get(c, self.handle_data)(c)

    def handle_desc_start(self, c):
        if self.state == 'BODY':
            self.state = 'DESC'
        else:
            self.handle_data(c)

    def handle_desc_end(self, c):
        if self.state == 'DESC':
            self.state = 'LINK'
        else:
            self.handle_data(c)

    def handle_url_start(self, c):
        if self.state == 'LINK':
            self.state = 'URL'
        else:
            self.handle_data(c)

    def handle_url_end(self, c):
        if self.state == 'URL':
            # Create a wikilink and append it to the results.
            self.results.append('[[')
            if self.urlBuffer:
                urlStr = ''.join(self.urlBuffer)
                urlStr = urlStr.removeprefix('./')
                urlStr = unquote(urlStr)
                self.results.append(urlStr)
                if self.descBuffer:
                    self.results.append('|')
                    self.results.extend(self.descBuffer)
            else:
                # Turn the description into an URL.
                urlStr = ''.join(self.descBuffer)
                urlStr = urlStr.replace(':', '/')
                urlStr = unquote(urlStr)
                self.results.append(urlStr)
            self.results.append(']]')
            self.urlBuffer.clear()
            self.descBuffer.clear()
            self.state = 'BODY'
        else:
            self.handle_data(c)

    def handle_data(self, c):
        if self.state == 'DESC':
            self.descBuffer.append(c)
            return

        if self.state == 'URL':
            self.urlBuffer.append(c)
            return

        if self.state == 'LINK':
            # Expected '(', but got another character:
            # the bracketed text is not a link description, so restore the body text.
            self.results.append('[')
            self.results.extend(self.descBuffer)
            self.results.append(']')
            self.descBuffer.clear()
            self.state = 'BODY'
        self.results.append(c)

    def close(self):
        """Append all buffered data to the results."""
        if self.descBuffer:
            self.results.append('[')
            self.results.extend(self.descBuffer)
        if self.urlBuffer:
            self.results.append('](')
            self.results.extend(self.urlBuffer)
        # incomplete Markdown links are adopted unchanged

