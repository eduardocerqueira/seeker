#date: 2025-01-06T16:34:29Z
#url: https://api.github.com/gists/556cca9004dc28dae66b746435fd4b3d
#owner: https://api.github.com/users/WiseScripts

#!/usr/bin/env python
#
# A script to check Kimsufi server availability using OVH's API.
# Typical usage:
#   python kimsufi.py --dump-txt --dump-json --loop --verbose 160sk1
#

import sys
import os.path
import argparse
import datetime
import time
import urllib.request
import pprint
import json

# kimsufi api
KSAPI_URL = "https://ws.ovh.com/dedicated/r2/ws.dispatcher/getAvailability2"
KSAPI_REQUESTS_QUOTA_SLEEP = 7.5 # requests quota (500 per 3600 seconds as of 2016-02)

# default paths to dump files
DUMPFILE_TXT = os.path.join(os.path.dirname(__file__), "kimsufi-response.log.txt")
DUMPFILE_JSON = os.path.join(os.path.dirname(__file__), "kimsufi-response.log.py")

# global vars
ctx = {}


def info(*args, **kwargs):
    if ctx['args'].verbose:
        print(*args, **kwargs)

def dump_txt(data):
    if ctx['args'].dump_txt:
        path = ctx['args'].dump_txt
    else:
        path = DUMPFILE_TXT
    with open(path, mode="wb") as f:
        f.write("# {}\n".format(datetime.datetime.now()).encode(encoding="utf-8"))
        f.write(data)

def dump_json(obj):
    if ctx['args'].dump_json:
        path = ctx['args'].dump_json
    else:
        path = DUMPFILE_JSON
    with open(path, mode="wt") as f:
        print("#", str(datetime.datetime.now()), file=f)
        print("json = ", end="", file=f)
        pprint.pprint(obj, stream=f, indent=2)

def get_availability():
    # perform request
    start_time = datetime.datetime.now()
    info("REQUEST #{}: {}...".format(ctx['request_id'], start_time), end="")
    try:
        with ctx['url_opener'].open(KSAPI_URL, timeout=KSAPI_REQUESTS_QUOTA_SLEEP + 1) as conn:
            raw_response = conn.read()
        info(" (elapsed {})".format(datetime.datetime.now() - start_time))
    except:
        info("")
        raise

    # do not dump before checking the response to ensure we parse the result as
    # fast as possible in case a reference turned out to be available since last
    # request
    try:
        # parse json
        json_response = None # ensure it is defined in the except block
        json_response = json.loads(raw_response.decode(encoding="utf-8", errors="strict"))

        # check for refs' availability
        available_refs = {}
        for element in json_response['answer']['availability']:
            if element['reference'].lower() in ctx['args'].refs:
                # here, we've found a matching server model reference
                server_ref = element['reference']
                for zone in element['zones']:
                    if zone['availability'].lower() not in ("unavailable", "unknown"):
                        # reference appears to be available
                        try:
                            available_refs[server_ref].append(zone['zone'])
                        except KeyError:
                            available_refs[server_ref] = [zone['zone']]

        return available_refs, raw_response, json_response
    except:
        dump_txt(raw_response)
        if json_response:
            dump_json(json_response)
        raise

def main():
    argp = argparse.ArgumentParser(
        description="Find the Kimsufi servers availability.")
    argp.add_argument("--dump-txt", metavar="TXTFILE", nargs="?", const=DUMPFILE_TXT,
        help="Dump the last answer from the API as-is.")
    argp.add_argument("--dump-json", metavar="PYFILE", nargs="?", const=DUMPFILE_JSON,
        help="Dump the last answer from the API JSON-then-Python-formated.")
    argp.add_argument("--loop", action="store_true",
        help="Loop until a server is available without exceeding requests quota.")
    argp.add_argument("--proxy", metavar="PROXY",
        help="Connect through the given HTTP(S) proxy URL.")
    argp.add_argument("--verbose", "-v", action="store_true",
        help="Print requests info.")
    argp.add_argument("refs", metavar="REFS", nargs="+",
        help="Model references (e.g. 160sk1 for KS-1, 160sk2 for KS-2A, ...)")
    ctx['args'] = argp.parse_args();

    if ctx['args'].proxy:
        ctx['url_opener'] = urllib.request.build_opener(
                                urllib.request.ProxyHandler({
                                    'http': ctx['args'].proxy,
                                    'https': ctx['args'].proxy}))
    else:
        ctx['url_opener'] = urllib.request.build_opener()

    ctx['request_id'] = 0
    while True:
        start_time = time.monotonic()
        ctx['request_id'] += 1

        available_refs, raw_response, json_response = get_availability()
        if available_refs:
            print("{} SERVER{} FOUND ({}):".format(
                                                len(available_refs),
                                                "S"[len(available_refs)==1:],
                                                datetime.datetime.now()))
            for ref, zones in available_refs.items():
                print("  {}: {}".format(ref, ", ".join(zones)))

        if ctx['args'].dump_txt:
            dump_txt(raw_response)
        if ctx['args'].dump_json:
            dump_json(json_response)

        if available_refs or not ctx['args'].loop:
            break

        # wait, but not more than we should!
        seconds_to_wait = KSAPI_REQUESTS_QUOTA_SLEEP - (time.monotonic() - start_time)
        if seconds_to_wait > 0:
            time.sleep(seconds_to_wait)

    return 0

if __name__ == "__main__":
    sys.exit(main())
