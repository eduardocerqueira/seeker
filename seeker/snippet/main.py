#date: 2024-02-27T17:03:16Z
#url: https://api.github.com/gists/b474b99082d33f36dd4e769c36f37a1a
#owner: https://api.github.com/users/mypy-play

class Category:
    """Your docstring.

    Attributes:
        text(str): a description.
    """

    text: str


a = Category()

Category.text = 11  # where is the warning?
a.text = 1.5454654  # where is the warning?
