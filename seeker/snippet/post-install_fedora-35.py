#date: 2022-02-07T16:51:46Z
#url: https://api.github.com/gists/d36533ddb67f766af9978e824c3d3e3a
#owner: https://api.github.com/users/arslivinski

#!/usr/bin/env python

import collections
import concurrent.futures
import enum
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request

# Glyphs
SPINNER = ["⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽", "⣾"]
GROUP = '●'
SUCESS = '✓'
FAILURE = '✕'
SKIPPED = '🠒'

# Format
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
DEFAULT = '\033[39m'


# Globals
resume = 0
step = 0


# TODO
# -[ ] Log
# -[ ] Bail on upgrade failure
# -[ ] Configure the container registries
# -[ ] Make zsh the default shell "chsh -s $(which zsh)"
# -[ ] Install volta.sh
# -[ ] Install deno


def main():
    print(BOLD, BLUE)
    print('█████▒  █████▒  █████▒  █▒  █▒  █████▒')
    print('█▒      █▒        █▒    █▒  █▒  █▒  █▒')
    print('█████▒  █████▒    █▒    █▒  █▒  █████▒')
    print('    █▒  █▒        █▒    █▒  █▒  █▒    ')
    print('█████▒  █████▒    █▒    █████▒  █▒    ')
    print(DEFAULT)
    print('                            Fedora 35 ')
    print(RESET)

    check_for_sudo()
    start_time = time.time()

    run_tasks([
        Group('Configure repositories', [
            Task('RPM Fusion - Free', repo_rpmfusion_free),
            Task('RPM Fusion - NonFree', repo_rpmfusion_nonfree),
        ]),
        Group('Update', [
            Task('Check for updates', dnf_check_update),
            Task('Update Appstream data', update_appstream),
            Task('Upgrade OS', dnf_upgrade),
#            Task('Reboot...', reboot),
        ]),
        Group('Install packages', [
            Task('aria2', dnf_install, 'aria2'),
            Task('bat', dnf_install, 'bat'), # Replacement for 'cat'
            Task('exa', dnf_install, 'exa'), # Replacement for 'ls'
            Task('fd', dnf_install, 'fd-find'), # Replacement for 'find'
            Task('Font: Fira Code', dnf_install, 'fira-code-fonts'),
            Task('Git Credential Libsecret', dnf_install, 'git-credential-libsecret'),
            Task('GNOME Extensions', dnf_install, 'gnome-extensions-app'),
            Task('GNOME Shell Extension AppIndicator', dnf_install, 'gnome-shell-extension-appindicator'),
            Task('GNOME Todo', dnf_install, 'gnome-todo'),
            Task('GNOME Tweaks', dnf_install, 'gnome-tweaks'),
            Task('Google Chrome', dnf_install_http, 'https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm'),
            Task('libatomic', dnf_install, 'libatomic'), # Discord dep
            Task('OpenSSL', dnf_install, 'openssl'),
            Task('Powerline', dnf_install, 'powerline'),
            Task('Powerline Fonts', dnf_install, 'powerline-fonts'),
            Task('Sushi', dnf_install, 'sushi'), # Quick preview for Nautilus
            Task('Tilix', dnf_install, 'tilix'),
            Task('UnRAR', dnf_install, 'unrar'), # Requires RPM Fusion NonFree
            Task('util-linux-user', dnf_install, 'util-linux-user'), # To use 'chsh'
            Task('Visual Studio Code', dnf_install_http, 'https://update.code.visualstudio.com/latest/linux-rpm-x64/stable'),
            Task('vim enhanced', dnf_install, 'vim-enhanced'),
            Task('VLC', dnf_install, 'vlc'), # Requires RPM Fusion
            Task('zsh', dnf_install, 'zsh'),
        ]),
        Group('Install package groups', [
            Task('C Development Tools and Libraries', dnf_group_install, 'C Development Tools and Libraries'),
        ]),
        Group('Install Flapak packages', [
            Group('Configure flatpak repositories', [
                Task('Flathub', add_flatpak_repo, 'flathub', 'https://flathub.org/repo/flathub.flatpakrepo'),
            ]),
#            Task('Apostrophe', flatpak_install, 'flathub', 'org.gnome.gitlab.somas.Apostrophe'),
            Task('Dialect', flatpak_install, 'flathub', 'com.github.gi_lom.dialect'),
            Task('Discord', flatpak_install, 'flathub', 'com.discordapp.Discord'),
            Task('Drawing', flatpak_install, 'flathub', 'com.github.maoschanz.drawing'),
            Task('Extension Manager', flatpak_install, 'flathub', 'com.mattjakeman.ExtensionManager'),
            Task('GIMP', flatpak_install, 'flathub', 'org.gimp.GIMP'),
            Task('GNOME Firmware', flatpak_install, 'flathub', 'org.gnome.Firmware'),
            Task('Flatseal', flatpak_install, 'flathub', 'com.github.tchx84.Flatseal'),
            Task('Inkscape', flatpak_install, 'flathub', 'org.inkscape.Inkscape'),
            Task('Insomnia', flatpak_install, 'flathub', 'rest.insomnia.Insomnia'),
            Task('Kooha', flatpak_install, 'flathub', 'io.github.seadve.Kooha'),
#            Task('Postman', flatpak_install, 'flathub', 'com.getpostman.Postman'),
            Task('Slack', flatpak_install, 'flathub', 'com.slack.Slack'),
            Task('Spotify', flatpak_install, 'flathub', 'com.spotify.Client'),
            Task('Telegram', flatpak_install, 'flathub', 'org.telegram.desktop'),
            Task('Video Trimmer', flatpak_install, 'flathub', 'org.gnome.gitlab.YaLTeR.VideoTrimmer'),
        ]),
        Group('Install codecs', [
            Task('GStreamer', install_gstreamer),
#            Task('Lame', install_lame),
#            Task('Multimedia', install_multimedia),
        ]),
        Group('Install GNOME Shell extensions', [ 
            Task('Removable Drive Menu', gnome_shell_extension_install, 7),
            Task('Remove Alt+Tab Delay v2', gnome_shell_extension_install, 2741),
            Task('Window Is Ready - Notification Remover', gnome_shell_extension_install, 1007),
        ]),
#        Group('Manual installations', [
#            Task('Deno', ), # TODO
#            Task('volta.sh', ), # TODO
#        ]),
        Group('Configuration', [
            Task('Create document template', create_template),
            Task('Hide folders on Nautilus', hide_folders),
            Task('Increase file watch', increase_file_watch),
            Task('Default DNF to Yes', dnf_default_yes),
#            Task('Remove unused container registries', remove_container_registries),
        ]),
        Group('Cleanup', [
            Task('dnf autoremove', run, 'dnf', '-qy', 'autoremove'),
            Task('dnf clean all', run, 'dnf', '-qy', 'clean', 'all'),
            Task('Resume script', run, 'rm', '-f', get_resume_script()),
        ]),
    ])

    elapsed_time = time.time() - start_time
    duration = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f'\nTotal elapsed time: {duration}')


def check_for_sudo():
    if os.environ.get('SUDO_USER') is None or os.geteuid() != 0:
        raise Exception('You should run this script with sudo!')


def repo_rpmfusion_free():
    dnf_install('https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-35.noarch.rpm')


def repo_rpmfusion_nonfree():
    dnf_install('https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-35.noarch.rpm')


def dnf_check_update():
    run('dnf', 'check-update', '--refresh')


def dnf_upgrade():
    run('dnf', 'upgrade', '-qy')


def update_appstream():
    run('dnf', 'groupupdate', '-qy', 'core')


def install_gstreamer():
    dnf_install(
        '--exclude=gstreamer1-plugins-bad-free-devel',
        'gstreamer1-plugins-base',
        'gstreamer1-plugins-good-*',
        'gstreamer1-plugins-bad-*',
        'gstreamer1-plugin-openh264',
        'gstreamer1-libav',
    )


def install_lame():
    run('dnf', 'install', '-qy', '--exclude=lame-devel', 'lame*')


def install_multimedia():
    run('dnf', 'group', 'upgrade', '-qy', '--with-optional', 'Multimedia')


def create_template():
    home = get_home_dir()
    file_write(f'{home}/Templates/text.txt', '', root=False)
    file_write(f'{home}/Templates/doc.md', '', root=False)
    file_write(f'{home}/Templates/script.js', '', root=False)


def hide_folders():
    home = get_home_dir()
    thigs_to_hide = f'''\
        Desktop
        Public
        Templates
    '''
    file_write(f'{home}/.hidden', thigs_to_hide, root=False)


def increase_file_watch():
    file_append('/etc/sysctl.conf', 'fs.inotify.max_user_watches=524288')


def dnf_default_yes():
    file_append('/etc/dnf/dnf.conf', 'defaultyes=True')


def remove_container_registries():
    # sed -z 's/\(\[registries\.search\]\nregistries = \[\)[^]]*\(\]\)/\1'\''docker.io'\''\2/' -i /etc/containers/registries.conf
    # sed -i "s/unqualified-search-registries = \[.*\]/unqualified-search-registries = \['docker.io'\]/" /etc/containers/registries.conf
    pass # TODO


def dnf_install(*args):
    run('dnf', 'install', '-qy', *args)


def dnf_install_http(url, *args):
    url_effective = run('curl', '-ILs', '-o', '/dev/null', '-w', '%{url_effective}', url)
    run('dnf', 'install', '-qy', *args, url_effective)


def dnf_group_install(*args):
    run('dnf', 'group', 'install', '-qy', *args)


def add_flatpak_repo(id, url):
    run('flatpak', 'remote-add', '--if-not-exists', id, url)


def flatpak_install(repository, id):
    run('flatpak', 'install', '-y', '--noninteractive', repository, id)


def dconf(key, value):
    try:
        run('dconf', 'write', key, value, sudo=False)
    except:
        pass


def natural_sort(l): 
    # https://stackoverflow.com/a/4836734
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def gnome_shell_extension_install(id):
    with urllib.request.urlopen(f'https://extensions.gnome.org/extension/{id}') as response:
        html = response.read().decode('utf-8')
        data_uuid = re.findall(r'data-uuid="([^"]+)"', html, re.MULTILINE)[0]
        data_svm = re.findall(r'data-svm="([^"]+)"', html, re.MULTILINE)[0]
        uuid = data_uuid.replace('@', '')
        svm = json.loads(data_svm.replace('&quot;', '"'))
        versions = natural_sort(list(svm.keys()))
        versions.reverse()
        latest = svm[versions[0]]['version']
        extension_file = f'{uuid}.v{latest}.shell-extension.zip'
        extension_url = f'https://extensions.gnome.org/extension-data/{extension_file}'

    try:
        response = urllib.request.urlopen(extension_url)
    except:
        uuid = data_uuid.replace('@', '%40')
        extension_file = f'{uuid}.v{latest}.shell-extension.zip'
        extension_url = f'https://extensions.gnome.org/extension-data/{extension_file}'
        response = urllib.request.urlopen(extension_url)

    with response:
        tmp_file = os.path.join(tempfile.gettempdir(), extension_file)
        with open(tmp_file, 'wb') as f:
            shutil.copyfileobj(response, f)
        chown_user(tmp_file)
        run('gnome-extensions', 'install', '-f', tmp_file, sudo=False)
        run('gnome-extensions', 'enable', data_uuid, sudo=False)


def run(*cmd, sudo=True):
    if not sudo:
        user = os.environ.get('SUDO_USER')
        cmd = ['sudo', '-u', user, *cmd]
    proc = subprocess.run(cmd, capture_output=True, encoding='utf-8')
    proc.check_returncode()
    return proc.stdout


def file_write(path, text, root=True):
    with open(path, mode='w', encoding='utf-8') as f:
        print(textwrap.dedent(text), file=f, flush=True)
    if not root:
        chown_user(path)


def file_append(path, text):
    with open(path, mode='a', encoding='utf-8') as f:
        print(textwrap.dedent(text), file=f, flush=True)


def reboot():
    input(f'\nPress {BOLD}Enter{RESET} to continue...')
    global resume
    global step
    resume_from = step + 1
    resume_script = get_resume_script()
    autostart = os.path.dirname(resume_script)
    os.makedirs(autostart, exist_ok=True)
    chown_user(autostart)
    file_write(resume_script, f'''\
        [Desktop Entry]
        Name=Resume Setup
        Comment=Resume the Setup script
        Exec=sudo python {sys.argv[0]} {resume_from}
        Encoding=UTF-8
        Type=Application
        Terminal=true
        StartupNotify=false
        OnlyShowIn=GNOME
        X-GNOME-Autostart-enabled=true
    ''', root=False)
    os.chmod(resume_script, 0o775)
    os.system('reboot')


def chown_user(path):
    uid = int(os.environ.get('SUDO_UID'))
    gid = int(os.environ.get('SUDO_GID'))
    os.chown(path, uid, gid)


def get_home_dir():
    user = os.environ.get('SUDO_USER')
    return f'/home/{user}'


def get_resume_script():
    return f'{get_home_dir()}/.config/autostart/setup.desktop'


class Group:
    def __init__(self, desc, tasks):
        self.desc = desc
        self.tasks = tasks


class Task:
    def __init__(self, desc, fn, *args, **kwargs):
        self.desc = desc
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


def run_tasks(tasks, depth=0):
    indent = '  ' * depth

    for task in tasks:
        if isinstance(task, Group):
            print(flush=True)
            print(f'{indent}{BOLD}{GROUP} {task.desc}{RESET}', flush=True)
            run_tasks(task.tasks, depth + 1)
            print(flush=True)
            continue

        global resume
        global step
        step = step + 1

        if step < resume:
            print(f'{indent}{DIM}{SKIPPED} {task.desc}{RESET}', flush=True)
            continue

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task.fn, *task.args, **task.kwargs)
            for frame in itertools.cycle(SPINNER):
                if future.done():
                    break
                icon = f'{BOLD}{BLUE}{frame}{RESET}'
                print(f'{indent}{icon} {task.desc}', end='\r', flush=True)
                time.sleep(1.0 / len(SPINNER))
            try:
                future.result()
            except:
                status = f'{BOLD}{RED}{FAILURE}{RESET}'
            else:
                status = f'{BOLD}{GREEN}{SUCESS}{RESET}'
            finally:
                print(f'{indent}{status} {task.desc}', flush=True)


def hide_cursor():
    print('\033[?25l', flush=True)


def show_cursor():
    print('\033[?25h', flush=True)


if __name__ == '__main__':
    try:
        if len(sys.argv) == 2:
            resume = int(sys.argv[1])
        hide_cursor()
        main()
    except Exception as exception:
        print(f'\n{RED}{exception}{RESET}')
    finally:
        if resume > 0:
            input(f'\nPress {BOLD}Enter{RESET} to continue...')
        show_cursor()
