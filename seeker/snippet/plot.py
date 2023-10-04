#date: 2023-10-04T17:07:57Z
#url: https://api.github.com/gists/3fe1435edb656964d8066b5b7da9af5f
#owner: https://api.github.com/users/amtoine

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"
NB_NS_IN_MS = 1_000_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("-s", "--skip-timestamps", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(args.data)

    build_times = [datetime.strptime(row["build_time"], DATE_FORMAT) for row in data]
    timestamps = [build_time.timestamp() for build_time in build_times]

    commits = [row["commit"] for row in data]
    avgs = np.array([row["avg"] / NB_NS_IN_MS for row in data])
    stddevs = np.array([row["stddev"] / NB_NS_IN_MS for row in data])

    fig, ax = plt.subplots(nrows=1, sharex=True)
    ax.plot(timestamps, avgs, color="blue", marker="o", label="mean startup time")
    ax.fill_between(
        timestamps,
        avgs + stddevs,
        avgs - stddevs,
        alpha=0.15,
        color="blue",
        label="standard deviation"
    )
    ax.set_title("startup times for each build of Nushell")

    plt.xticks(
        timestamps[::args.skip_timestamps + 1],
        [
            build_time.strftime("%Y-%m-%d") + f"\n{commit[:7]}"
            for (build_time, commit) in zip(build_times, commits)
        ][::args.skip_timestamps + 1]
    )
    plt.xlabel("build (date and commit)")
    plt.ylabel("time (in ms)")

    plt.grid(True)
    plt.legend()

    plt.show()