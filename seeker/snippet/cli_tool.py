#date: 2025-08-08T16:52:36Z
#url: https://api.github.com/gists/cace78ba2d9f5c191db90ab8c1413239
#owner: https://api.github.com/users/amalks4

import click

@click.command(help="this is just a hello app. it does nothing")
@click.option("--name", prompt="I need your name", help="Need name")
@click.option("--color", prompt="I need your color", help="this is your color")
def hello(name,color):
    if name == "thor":
        click.echo("thor you are always red")
        click.echo(click.style(f"Hello World! {name}",fg="red"))
    else:
        click.echo(click.style(f"Hello World! {name}",fg=color))

if __name__ == '__main__':
    hello()