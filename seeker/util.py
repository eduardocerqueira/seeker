import logging
import json
import re
from configparser import SafeConfigParser
from datetime import datetime, timedelta
from os import remove, rename, listdir
from subprocess import call, check_output


def get_config(section, parameter):
    config = SafeConfigParser()
    config.read("seeker.conf")
    return json.loads(config.get(section, parameter))


def prepend_line(file_name, line):
    dummy_file = file_name + ".bak"
    with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
        write_obj.write(line + "\n")
        for line in read_obj:
            write_obj.write(line)
    remove(file_name)
    rename(dummy_file, file_name)


def git_status(now):
    rs = check_output(["git", "status"], universal_newlines=True)
    line = "-" * 80
    report_header = f"{line}\n " f"{now}\n" f"{line}\n "
    prepend_line("report.txt", f"{report_header} {rs}")


def git_push():
    now = datetime.now()
    git_status(now)
    commit_message = f"{now} new snippets"
    # call("git pull origin main", shell=True)
    call("git add .", shell=True)
    call('git commit -m "' + commit_message + '"', shell=True)
    call("git push origin main", shell=True)


def purge():
    day = get_config("purge", "day")
    files = listdir("snippet")
    for file in files:
        with open(f"snippet/{file}", "r") as fp:
            data = fp.read()
            re_pattern = "(date:).(\d{4}-\d{2}-\d{2})"
            for m in re.finditer(re_pattern, data):
                if m.group(0).__len__() > 0:
                    date_in_header = datetime.strptime(
                        m.group(0).replace("date: ", ""), "%Y-%m-%d"
                    )
                    dt = datetime.today() - timedelta(days=day)
                    if date_in_header < dt:
                        logging.info(
                            f"header: {date_in_header} deleting snippet/{file}"
                        )
                        remove(f"snippet/{file}")


def build_regex():
    """
    build regex from obfuscate_rules
    :return: str with regex for capturing sensitive data
    """

    rules_content = get_config("obfuscate_rules", "content")
    rules_content_size = len(rules_content)

    re_pattern = []
    re_open = "(.*"
    re_close = ".*)"
    re_join = "|"

    for item in rules_content:
        index = rules_content.index(item)
        if index == rules_content_size - 1:
            re_join = ""
        re_pattern.append(f"{re_open}{item}{re_close}{re_join}")

    return "".join(re_pattern)


def obfuscate():
    rules_separator = get_config("obfuscate_rules", "separator")
    rules_mask = get_config("obfuscate_rules", "mask")
    re_pattern = build_regex()

    files = listdir("snippet")
    for file in files:
        with open(f"snippet/{file}", "r+") as fp:
            data = fp.read()
            sensitive_data = re.finditer(re_pattern, data, re.IGNORECASE)
            to_replace = {}
            for match_obj in sensitive_data:
                # extract separator from seeker.conf obfuscate_rules
                for separator in rules_separator:
                    if separator in match_obj.group():
                        sensitive_cred_key = match_obj.group().split(separator)[0]
                        sensitive_cred_value = match_obj.group().split(separator)[1]
                        before = match_obj.group()

                        logging.info(
                            f"Found sensitive data in {file} on {sensitive_cred_key}"
                        )

                        if sensitive_cred_value.replace('"', "").strip() == rules_mask:
                            logging.info(
                                f"secret {sensitive_cred_key} is already masked"
                            )
                            break

                        after = before.replace(sensitive_cred_value, f' "{rules_mask}"')
                        to_replace[before] = after
                        break

            if len(to_replace) > 0:
                fp.seek(0)

                for k, v in to_replace.items():
                    data = data.replace(k, v)

                fp.write(data)
