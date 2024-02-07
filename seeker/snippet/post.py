#date: 2024-02-07T16:57:32Z
#url: https://api.github.com/gists/94bdd5f62d22bf73fd0ac12f2ece3945
#owner: https://api.github.com/users/ZoroMNastya

from model.user import User


class Post:

    def __int__(self, body: str, author: User):
        self.body = body
        self.author = author
