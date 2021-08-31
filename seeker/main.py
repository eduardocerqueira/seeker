from provider import Gists
from seeker.util import git_push, purge

if __name__ == '__main__':
    print("Seeker is running ...")
    g = Gists()
    # life cycle
    g.get()
    purge()
    git_push()
