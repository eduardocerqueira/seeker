import click
import logging
from seeker.provider import Gists
from seeker.util import git_push, purge, obfuscate

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def run():
    logging.info("Seeker is running ...")
    g = Gists()
    # life cycle
    logging.info("purging old snippets")
    purge()
    logging.info("pushing to repo")
    git_push()
    # logging.info("getting new snippets")
    # g.get()
    # logging.info("obfuscating sensitive data")
    # obfuscate()
    logging.info("pushing to repo")
    git_push()


@click.command()
@click.option("--test", is_flag=True, help="test seeker app")
def cli(test):
    if test:
        click.echo("seeker is fine!")
        exit(0)
    run()


if __name__ == "__main__":
    cli()
