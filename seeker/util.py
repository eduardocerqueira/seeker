from configparser import SafeConfigParser
import json
from subprocess import call
from datetime import datetime


def get_config(section, parameter):
    config = SafeConfigParser()
    config.read("seeker.conf")
    return json.loads(config.get(section, parameter))


def git_status(now):
    rs = call("status", shell=True)
    line = "-" * 80
    report_header = f"{line}\n " \
                    f"{now}\n" \
                    f"{line}\n "
    with open("report.txt", "a") as report:
        data = f"{report_header} {rs}"
        report.write(data)


def git_push():
    now = datetime.now()
    git_status(now)
    commit_message = f"{now} new snippets"
    call("git add snippet", shell=True)
    call('git commit -m "' + commit_message + '"', shell=True)
    call('git push origin main', shell=True)
