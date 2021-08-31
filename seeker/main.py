from provider import Gists
from seeker.util import git_push

if __name__ == '__main__':
    print("Seeker")
    g = Gists()
    g.get()
    git_push()
