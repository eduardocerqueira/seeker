#date: 2023-07-20T17:07:15Z
#url: https://api.github.com/gists/e17b0bea9eae60c615d5758ac187d243
#owner: https://api.github.com/users/Urgau

#!/usr/bin/env python3

import statistics
import json
import sys
import os

USAGE = """USAGE
    ./print-vtable-sizes-summary FILE
DESCRIPTION
    FILE  JSON-ish file of the output of rustc with -Zprint-vtable-sizes"""

def without(filepath):
    return os.path.splitext(filepath)[0]

def upcasting_costs(mode, costs):
    upcasting_cost_percent_min = min(costs)
    upcasting_cost_percent_max = max(costs)
    upcasting_cost_percent_mean = statistics.fmean(costs)
    upcasting_cost_percent_stdev = statistics.stdev(costs)
    
    print(f"Upcasting cost ({mode}): ~{upcasting_cost_percent_mean:.3}% ± {upcasting_cost_percent_stdev:.3} (min: {upcasting_cost_percent_min:.3}, max: {upcasting_cost_percent_max:.3})")

def summary(filepath):
    prints = []
    with open(filepath, "r") as f:
        for l in f.readlines():
            if l:
                prints.append(json.loads(l.lstrip("print-vtable-sizes").strip()))

    total_entries = sum([int(p["entries"]) for p in prints])
    total_entries_for_upcasting = sum([int(p["entries_for_upcasting"]) for p in prints])

    upcasting_cost_percent_if_any = []
    upcasting_cost_percent_globaly = []
    for p in prints:
        percent = float(p["upcasting_cost_percent"])
        if int(p["entries_for_upcasting"]) > 0:
            upcasting_cost_percent_if_any.append(percent)
        upcasting_cost_percent_globaly.append(percent)

    print(f"## {without(filepath)} (with dependencies)")
    print()
    print(f"Total entries: {total_entries} (for {len(prints)} prints)")
    print(f"Total entries for upcasting: {total_entries_for_upcasting} (~{total_entries_for_upcasting / total_entries * 100:.2}%)")
    print()
    upcasting_costs("if any", upcasting_cost_percent_if_any)
    upcasting_costs("globably", upcasting_cost_percent_globaly)

def entry():
    if len(sys.argv) == 2 and (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
        print(USAGE)
        exit(0)
    if len(sys.argv) < 2:
        raise Exception("Invalid number of argruments")

    filepaths = sys.argv[1:]

    summary_for = ", ".join(without(filepath) for filepath in filepaths)
    print(f"# Summary for {summary_for}")
    print("")
    print("*Upcasting cost (MODE): MEAN ± STDEV (MIN, MAX)*")

    for filepath in filepaths:
        print()
        summary(filepath)

if __name__ == "__main__":
    try:
        entry()
    except Exception as e:
        print(e, file=sys.stderr)
        exit(84)