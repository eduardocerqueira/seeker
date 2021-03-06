from configparser import SafeConfigParser
import json
from subprocess import call, check_output
from datetime import datetime, timedelta
from os import remove, rename, listdir


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
    rs = check_output(["git", "status"], universal_newlines=True)
    line = "-" * 80
    report_header = f"{line}\n " \
                    f"{now}\n" \
                    f"{line}\n "
    prepend_line("report.txt", f"{report_header} {rs}")


def git_push():
    now = datetime.now()
    git_status(now)
    commit_message = f"{now} new snippets"
    call("git add .", shell=True)
    call('git commit -m "' + commit_message + '"', shell=True)
    call('git push origin main', shell=True)


def purge():
    day = get_config("purge", "day")
    files = listdir("snippet")
    for file in files:
        with open(f"snippet/{file}", "r") as fp:
            data = fp.read()
            dt = datetime.today() - timedelta(days=day)
            if dt.strftime("%Y-%m-%d") in data:
                remove(f"snippet/{file}")
