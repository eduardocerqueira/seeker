#date: 2025-09-01T17:09:08Z
#url: https://api.github.com/gists/bccc52439cea57e1a4369a8c1b4bbc04
#owner: https://api.github.com/users/ubiquitousfire

class Post:
    def __init__(self, post_id, title, subtitle, body):
        self.id = post_id
        self.title = title
        self.subtitle = subtitle
        self.body = body