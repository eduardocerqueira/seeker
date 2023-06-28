#date: 2023-06-28T17:07:06Z
#url: https://api.github.com/gists/a52dec5e704619449937a6774e4f7bb1
#owner: https://api.github.com/users/rsiemens

"""
MIT License

Copyright (c) 2023 Ryan Siemens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import re
import string


 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********": "**********"
    AT = "TOK_@"
    L_BRACKET = "TOK_["
    R_BRACKET = "TOK_]"
    HASH = "TOK_#"
    CARET = "TOK_^"
    INT = "TOK_INT"
    WS_NEWLINE = "TOK_WS_NEWLINE"
    WS_NON_NEWLINE = "TOK_WS_NON_NEWLINE"
    WORD = "TOK_WORD"

    def __init__(self, tok_type: str, value: str):
        self.tok_type = tok_type
        self.value = value

    def __repr__(self):
        return f"<{self.tok_type} {repr(self.value)}>"

    def __eq__(self, other):
        return (
            isinstance(other, Token)
            and other.tok_type == self.tok_type
            and other.value == self.value
        )


class Lexer:
    PRINTABLE = set(string.printable) - set(string.whitespace) - set("[]@")

    def __init__(self, source: str):
        self.source = source
        self.cursor = 0

    @property
    def current(self):
        try:
            return self.source[self.cursor]
        except IndexError:
            return None

    @property
    def peek_next(self):
        try:
            return self.source[self.cursor + 1]
        except IndexError:
            return None

    def next(self):
        self.cursor += 1
        return self.current

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        if self.current == "@":
            return Token(Token.AT, self.current)
        elif self.current == "[":
            return Token(Token.L_BRACKET, self.current)
        elif self.current == "]":
            return Token(Token.R_BRACKET, self.current)
        elif self.current == "#":
            return Token(Token.HASH, self.current)
        elif self.current == "^":
            return Token(Token.CARET, self.current)
        elif self.current in string.digits:
            num = self.current
            while self.peek_next in string.digits:
                num += self.next()
            return Token(Token.INT, int(num))
        elif self.current == "\n":
            return Token(Token.WS_NEWLINE, self.current)
        elif self.current in " \t":
            whitespace = self.current
            while self.peek_next in " \t":
                whitespace += self.next()
            return Token(Token.WS_NON_NEWLINE, whitespace)
        elif self.current in self.PRINTABLE:
            word = self.current
            while self.peek_next in self.PRINTABLE:
                word += self.next()
            return Token(Token.WORD, word)
        raise Exception(
            f"Unexpected character ({self.current}) @ position {self.cursor} when lexing"
        )

    def lex(self):
        while self.current:
            yield self.tokenize()
            self.next()


class Parser:
    def __init__(self, source: str):
        self.source = source
        self.tokens = "**********"
        self.current = "**********"

    def next(self):
        try:
            self.current = "**********"
        except StopIteration:
            self.current = None
        return self.current

    def expect(self, tok_type):
        if self.current.tok_type != tok_type:
            raise Exception(f"expected {tok_type}, got {self.current}")

    def parse(self):
        return self.codeowners()

    def codeowners(self):
        current_section = Section("", [])  # the default section
        sections = [current_section]
        while self.current:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"L "**********"_ "**********"B "**********"R "**********"A "**********"C "**********"K "**********"E "**********"T "**********", "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"C "**********"A "**********"R "**********"E "**********"T "**********"] "**********": "**********"
                current_section = self.section()
                sections.append(current_section)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"O "**********"R "**********"D "**********": "**********"
                current_section.entries.append(self.entry(current_section))
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"H "**********"A "**********"S "**********"H "**********": "**********"
                self.eat_comment()
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"S "**********"_ "**********"N "**********"O "**********"N "**********"_ "**********"N "**********"E "**********"W "**********"L "**********"I "**********"N "**********"E "**********": "**********"
                self.eat_whitespace()
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"S "**********"_ "**********"N "**********"E "**********"W "**********"L "**********"I "**********"N "**********"E "**********": "**********"
                self.next()
            else:
                breakpoint()
                raise Exception("unparsable")
        return CodeOwners(sections)

    def section(self):
        name = None
        optional = False
        required_approvals = 1

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"C "**********"A "**********"R "**********"E "**********"T "**********": "**********"
            optional = True
            self.next()

        self.expect(Token.L_BRACKET)
        self.next()
        self.eat_whitespace()
        self.expect(Token.WORD)
        name = self.current.value
        self.next()

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"  "**********"a "**********"n "**********"d "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"! "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"R "**********"_ "**********"B "**********"R "**********"A "**********"C "**********"K "**********"E "**********"T "**********": "**********"
            name += self.current.value
            self.next()
        self.next()

        self.eat_whitespace()

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"L "**********"_ "**********"B "**********"R "**********"A "**********"C "**********"K "**********"E "**********"T "**********": "**********"
            self.next()
            self.expect(Token.INT)
            required_approvals = self.current.value
            self.next()
            self.expect(Token.R_BRACKET)
            self.next()
            self.eat_whitespace()
    
        default_owners = self.owners()

        return Section(name, default_owners, optional, required_approvals)

    def entry(self, section):
        self.expect(Token.WORD)
        pattern = self.current.value
        self.next()

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"[ "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"A "**********"T "**********", "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"S "**********"_ "**********"N "**********"E "**********"W "**********"L "**********"I "**********"N "**********"E "**********"] "**********": "**********"
            pattern += self.current.value
            self.next()

        self.eat_whitespace()
        owners = self.owners()
        return Entry(pattern.strip(), section, owners)

    def owners(self):
        owners = []
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"  "**********"a "**********"n "**********"d "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"A "**********"T "**********": "**********"
            self.next()
            self.expect(Token.WORD)
            owners.append(f"@{self.current.value}")
            self.next()
            self.eat_whitespace()
        return owners

    def eat_comment(self):
        self.expect(Token.HASH)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"  "**********"a "**********"n "**********"d "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"! "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"S "**********"_ "**********"N "**********"E "**********"W "**********"L "**********"I "**********"N "**********"E "**********": "**********"
            self.next()
        self.next()

    def eat_whitespace(self):
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********"  "**********"a "**********"n "**********"d "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"c "**********"u "**********"r "**********"r "**********"e "**********"n "**********"t "**********". "**********"t "**********"o "**********"k "**********"_ "**********"t "**********"y "**********"p "**********"e "**********"  "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********". "**********"W "**********"S "**********"_ "**********"N "**********"O "**********"N "**********"_ "**********"N "**********"E "**********"W "**********"L "**********"I "**********"N "**********"E "**********": "**********"
            self.next()


class CodeOwners:
    def __init__(self, sections):
        self.sections = sections

    def find_owners(self, path):
        matches = {}
        for section in self.sections:
            for entry in section.entries:
                if re.match(entry.re_pattern, path) is not None:
                    # last pattern wins with repeats
                    matches[entry.pattern] = entry

        return [e for e in matches.values()]


class Section:
    def __init__(self, name, default_owners=None, optional=False, required_approvals=1):
        self.name = name
        self.entries = []
        self.default_owners = default_owners
        self.optional = optional

        # Optional sections ignore the number of approvals required.
        if optional:
            self.required_approvals = 0
        else:
            self.required_approvals = required_approvals

    def __repr__(self):
        return f'<Section "{self.name}" {self.default_owners}>'


class Entry:
    def __init__(self, pattern, section, owners=None):
        self.pattern = pattern
        self.section = section
        self.owners = owners
        self.re_pattern = self._re_pattern(pattern)

    def _re_pattern(self, pattern):
        re_pattern = pattern
        re_pattern = re_pattern.replace(".", "\.")
        re_pattern = re_pattern.replace("**", "__db_star__")
        re_pattern = re_pattern.replace("*", "[^/]+")
        re_pattern = re_pattern.replace("__dbl_star__", ".+")
        if not pattern.startswith("/"):
            re_pattern = "/.+/" + re_pattern
        if pattern.endswith("/"):
            re_pattern += ".+"
        return "^" + re_pattern + "$"

    def __repr__(self):
        return f'<Entry "{self.pattern}" {self.owners}>'
