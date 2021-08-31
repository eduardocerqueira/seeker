from provider import Gists
from seeker.util import git_push, purge
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    logging.info("Seeker is running ...")
    g = Gists()
    # life cycle
    logging.info("getting new snippets")
    g.get()
    logging.info("purging old snippets")
    purge()
    logging.info("pushing to repo")
    git_push()
