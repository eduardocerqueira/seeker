#date: 2022-09-29T17:13:55Z
#url: https://api.github.com/gists/bcacde59d8824637ca18d90eed7711b2
#owner: https://api.github.com/users/schlagenhauf

import re

## Example log format and access log line
log_format = '$host $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent"'
line = 'sub.domain.net 123.123.123.123 - tux [27/Sep/2022:15:09:53 +0000] "GET /api/endpoint_groups HTTP/2.0" 200 198 "https://sub.domain.net/" "Mozilla/5.0 (X11; Linux x86_64; rv:105.0) Gecko/20100101 Firefox/105.0"'

## Part 1 - Build a regex pattern from the nginx log format
log_format = re.escape(log_format)                                # the nginx logging format contains special characters
field_matcher = re.compile(r'\\\$([A-Za-z0-9_]+)')                # match field names (alphanumeric and underscore)
field_names = field_matcher.findall(log_format)                   # first pass, collect all field names
regex_log_format = f"^{field_matcher.sub(r'(.+?)', log_format)}$" # second pass, replace field variables by a catch-all regex pattern

## Part 2 - Match all lines and extract capture groups
line_matcher = re.compile(regex_log_format)                       # We use this pattern for all lines, compile it first
line_match = line_matcher.match(line)                             # Match the whole line. None if the line does not fit the log_format
fields = dict(zip(field_names, line_match.groups()))              # Put it in a nice dict

print(fields)

# Prints:
# {'host': 'sub.domain.net', 'remote_addr': '123.123.123.123', 'remote_user': 'tux', 'time_local':
# '27/Sep/2022:15:09:53 +0000', 'request': 'GET /api/endpoint_groups HTTP/2.0', 'status': '200',
# 'body_bytes_sent': '198', 'http_referer': 'https://sub.domain.net/', 'http_user_agent': 'Mozilla/5.0
# (X11; Linux x86_64; rv:105.0) Gecko/20100101 Firefox/105.0'}