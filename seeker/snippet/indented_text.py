#date: 2023-02-20T16:40:01Z
#url: https://api.github.com/gists/0544ab89f77f8fd221bd1d1faf89b90e
#owner: https://api.github.com/users/angely-dev

class IndentedText:
    #
    # The given text must be indented by indent_char (default is " ")
    # and may have comments starting by comment_char (default is "#").
    #
    def __init__(self, text: str, indent_char: str = ' ', comment_char: str = '#'):
        if len(indent_char) != 1:
            raise ValueError(f'"indent_char" must be a char, not a str')
        if len(comment_char) != 1:
            raise ValueError(f'"comment_char" must be a char, not a str')
        self.text = text
        self.indent_char = indent_char
        self.comment_char = comment_char

    #
    # Return an n-ary tree representation of the indented text as a dict:
    #   - text is assumed to be correctly indented (see "sanitize" if needed)
    #   - lines are assumed to be unique per block (as they are used as keys)
    #
    # A KeyError may be raised if the text is not correctly indented.
    #
    def to_tree(self) -> {}:
        tree = {}

        # Last parents encountered indexed by their indentation level
        last_parent = {0: tree}

        for line in self.text.splitlines():
            child_name = line.lstrip(self.indent_char)
            child_level = len(line) - len(child_name)
            last_parent[child_level][child_name] = {}
            last_parent[child_level + 1] = last_parent[child_level][child_name]

        return tree

    #
    # Sanitize the indented text, meaning:
    #   - remove trailing spaces from lines, ignore blank and comment lines
    #   - ensure or fix indentation level (fix_indent being True by default)
    #
    # An IndentationError may be raised (indicating the faulty line) if fix_indent is False.
    #
    def sanitize(self, fix_indent: bool = True):
        sanitized_indented_text = []
        max_indent_level = 0 # max indentation level allowed (0 for the first line)
        line_num = 0         # maintain line number for error output purposes

        for line in self.text.splitlines():
            line_num += 1

            # line may have trailing spaces, remove them
            line = line.rstrip()

            # line may be blank or a comment, ignore it
            if not line or line.lstrip().startswith(self.comment_char):
                continue

            # ensure or fix line indentation level
            line_indent_level = len(line) - len(line.lstrip(self.indent_char))

            if not line_indent_level <= max_indent_level:
                # line indentation level is NOT correct, either raise an error or fix it
                if not fix_indent:
                    raise IndentationError(f'Line #{line_num} "{line}" is badly indented')
                else:
                    line_indent_level = max_indent_level               # fix indent level
                    line = line.lstrip(self.indent_char)               # remove bad indent
                    line = line_indent_level * self.indent_char + line # prepend good indent

            max_indent_level = line_indent_level + 1

            # line is sanitized at this point
            sanitized_indented_text.append(line)

        # text has been sanitized
        self.text = '\n'.join(sanitized_indented_text)

    def __str__(self):
        return self.text