#date: 2023-04-21T16:46:48Z
#url: https://api.github.com/gists/fb8bd80f5e59ab9778d379ac5f427db1
#owner: https://api.github.com/users/kaijicode

# docker ps
# docker ps -a
# docker ps --all

# docker volume ls
# docker volume ls -f dangling=true
# docker volume ls -q



import argparse
parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(dest='command')

# docker start server
# docker start server -i
cmd_start = sub_parsers.add_parser('start')
cmd_start.add_argument('-a', '--attach', action='store_true', help='Attach STDOUT/STDERR and forward signals')
cmd_start.add_argument('--detach-keys', type=str, help='Override the key sequence for detaching a container')
cmd_start.add_argument('-i', '--interactive', action='store_true', help='Attach container\'s STDIN')
cmd_start.add_argument('container')


# docker stop server
# docker stop server -t 10
cmd_stop = sub_parsers.add_parser('stop')
cmd_stop.add_argument('-t', '--time', type=int, help='Seconds to wait for stop before killing it (default 10)')
cmd_stop.add_argument('container')


# docker container ls
# docker container ls -a
cmd_container = sub_parsers.add_parser('container')
cmd_container_sub_parser = cmd_container.add_subparsers(dest='sub_command')

cmd_container__ls = cmd_container_sub_parser.add_parser('ls')
cmd_container__ls.add_argument('-a', '--all', action='store_true', help='Show all containers (default shows just running)')
cmd_container__ls.add_argument('-f', '--filter', help='Filter output based on conditions provided')


# docker container logs
# docker container logs -f
cmd_container__logs = cmd_container_sub_parser.add_parser('logs')
cmd_container__logs.add_argument('--details', help='Show extra details provided to logs')
cmd_container__logs.add_argument('-f', '--follow', help='Follow log output')

# Example: Consuming rest of the arguments
# docker container exec server -- bash -c 'venv3/bin/python manage.py shell'
# Namespace(command='container', sub_command='exec', container='server', args=['bash', '-c', 'ps'])
cmd_container__exec = cmd_container_sub_parser.add_parser('exec')
cmd_container__exec.add_argument('container')
cmd_container__exec.add_argument('args', nargs='*')

args = parser.parse_args()
print(args)

# python3 docker.py --help