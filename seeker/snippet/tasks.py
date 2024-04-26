#date: 2024-04-26T16:49:44Z
#url: https://api.github.com/gists/5c459e6faf6c2e8d010ddfed4f7b7bc3
#owner: https://api.github.com/users/jakubgrad

import os 
from invoke import task
from subprocess import call
from sys import platform

root_dir_raw = os.path.dirname(os.path.abspath(__file__))
root_dir = "'"+root_dir_raw+"'"

@task
def start(ctx):
    ctx.run(f"python3 {root_dir}/src/main.py ", pty=True)

@task
def jps(ctx):
    ctx.run(f"cd {root_dir}/src && python3 cli.py --jps --visual --map wall.map 0 0 4 7", pty=True)

@task
def dijkstra(ctx):
    ctx.run(f"cd {root_dir}/src && python3 cli.py --dijkstra --map arena.map 4 3 5 11", pty=True)

@task
def test(ctx):
    ctx.run(f"cd {root_dir} && pytest {root_dir}/src", pty=True)

@task
def coverage(ctx):
    ctx.run(f"cd {root_dir} && coverage run --branch -m pytest src", pty=True)

@task(coverage)
def coverage_report(ctx):
    ctx.run(f"cd {root_dir} && coverage html", pty=True)
    if platform != "win32":
        call(("xdg-open", f"{root_dir_raw}/htmlcov/index.html"))

@task
def format(ctx):
    ctx.run(f"cd {root_dir} && autopep8 --in-place --recursive src", pty=True)
