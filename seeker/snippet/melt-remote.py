#date: 2023-11-07T16:46:21Z
#url: https://api.github.com/gists/f808fa209c5ce2e712c90fc803ea42e4
#owner: https://api.github.com/users/lucasoethe

# This is a very simple script to run a .mlt script on a remote server.
# The remote needs to be reachable via ssh and needs to have melt, tmux and xvfb-run installed.
# The script will analyse the .mlt file given and figure out which resources are used
# it will then give the user a list of these resources with the option to remap them.
# This is useful if some (or all) of the files are already present on the remote server.
# A file mapped to "None" is assumed to be not present on the server and will be copied over.
# These files are copied to a new root on the server, which is based on the --folder option.
# The path given via the --folder option is combined with a UUID to generate a per-project folder.
# The script will then give an output.mlt suitable to be run on the remote server and copy it
# , as well as all the resources mapped to "None" earlier, to the server.
# It will then start a new tmux session on the server in which it runs
# xvfb-run -a melt path/to/output.mlt

import xml.etree.ElementTree as ET
import uuid
import os
import argparse

parser = argparse.ArgumentParser(
    prog='melt-remote',
    description='Executes a melt script on a remote server, copying the files if necessary'
)

parser.add_argument('target', help='The target server to run on, will be passed to ssh and rsync')
parser.add_argument('input', help='The input melt file (Default: input.mlt)', nargs='?', default='input.mlt')
parser.add_argument('-f', '--folder', help='The folder on the remote server to use (Default: ~/render)', default='~/render')
parser.add_argument('-d', '--dry-run', help='Creates the output.mlt file locally, but does not execute any commands on the remote server', action='store_true')
args = parser.parse_args()

target = args.target
target_folder = args.folder
dry_run = args.dry_run
rsync_command = "rsync --info=progress2 "

tree = ET.parse('input.mlt')
root = tree.getroot()
new_uuid = str(uuid.uuid4())
new_root = os.path.join(target_folder, new_uuid)
old_root = root.attrib['root']
root.attrib['root'] = new_root
print(f"Root: {old_root} -> {new_root}")
consumers = root.findall('consumer')
if len(consumers) > 1:
    print("Warning: Multiple consumers")
i = 0
for consumer in consumers:
    old_target = consumer.attrib['target']
    new_target = os.path.join(new_root, f"out{i}.mp4")
    consumer.attrib['target'] = new_target
    i = i + 1
    print(f"Consumer: {old_target} -> {new_target}")

rsync_commands = []
resource_map = {}

for resource in root.findall('.//property[@name="resource"]'):
    old_name = resource.text
    if not old_name.startswith('/'):
        old_name = os.path.join(old_root, old_name)
    
    if os.path.exists(old_name):
        resource_map[old_name] = None
    else:
        print(f'{resource.text} not found, assuming to be none file resource')

while True:
    print()
    print("Mapping: ")
    temp = {}
    last_id = 0
    for id, resource in enumerate(resource_map):
        temp[id + 1] = resource
        print(f'({id + 1}) {resource} -> {resource_map[resource]}')
        last_id = id + 1
    last_id = last_id + 1
    print(f'({last_id}) Re-root all')
    choise = input("Which input to modify (empty to continue): ")
    if choise == "":
        break

    try:
        choise = int(choise)
    except:
        continue
    
    if choise < 1 or choise > last_id:
        continue
    
    if choise == last_id:
        new = input("Please enter a new root for all files: ")
        for resource in resource_map:
            basename = os.path.basename(resource)
            resource_map[resource] = os.path.join(new, basename)
    else:
        old = temp[choise]
        new = input(f"Please provide a new target path for {old} (empty for None): ")
        if new == "":
            resource_map[old] = None
        else:
            resource_map[old] = new


for resource in resource_map:
    if resource_map[resource] == None:
        _, extension = os.path.splitext(resource)
        u = uuid.uuid4()
        new_name = os.path.join(new_root, f'{u}{extension}')
        resource_map[resource] = new_name
        rsync_commands.append(f'{rsync_command} "{resource}" "{target}:{new_name}"')

for resource in root.findall('.//property[@name="resource"]'):
    old_name = resource.text
    if not old_name.startswith('/'):
        old_name = f'{old_root}/{old_name}'
    if os.path.exists(old_name):
        resource.text = resource_map[old_name]

tree.write('output.mlt')
rsync_commands.append(f'{rsync_command} "output.mlt" "{target}:{os.path.join(new_root, "output.mlt")}"')
command_list = [f'ssh {target} "mkdir -p {new_root}"'] + rsync_commands + [f'ssh {target} "tmux new-session -d -s melt-{new_uuid} \\"xvfb-run -a melt {os.path.join(new_root, "output.mlt")}\\""']
for command in command_list:
    print(f'> {command}')
    if not dry_run:
        os.system(command)