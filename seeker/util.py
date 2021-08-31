from configparser import SafeConfigParser
import json
from subprocess import call
from datetime import datetime
from os import remove, rename


def get_config(section, parameter):
    config = SafeConfigParser()
    config.read("seeker.conf")
    return json.loads(config.get(section, parameter))


def prepend_line(file_name, line):
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    remove(file_name)
    rename(dummy_file, file_name)


def git_status(now):
    rs = call("status", shell=True)
    line = "-" * 80
    report_header = f"{line}\n " \
                    f"{now}\n" \
                    f"{line}\n "
    prepend_line("report.txt", f"{report_header} {rs}")


def git_push():
    now = datetime.now()
    git_status(now)
    commit_message = f"{now} new snippets"
    call("git add", shell=True)
    call('git commit -m "' + commit_message + '"', shell=True)
    call('git push origin main', shell=True)
