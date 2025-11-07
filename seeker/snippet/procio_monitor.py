#date: 2025-11-07T17:08:28Z
#url: https://api.github.com/gists/84838b60b4f0f6959c13148aaa3d4640
#owner: https://api.github.com/users/hneiva

#!/usr/bin/env python3
"""
procio_monitor.py
A lightweight 'iotop-like' Python script to continuously list processes
with the highest disk read/write activity in bytes/sec over a sampling interval.

Requirements: psutil

Usage examples:
  python procio_monitor.py                # default: interval=1.0s, top 15, sort by write
  python procio_monitor.py --interval 2   # sample every 2 seconds
  python procio_monitor.py --sort total   # sort by total (read+write) throughput
  python procio_monitor.py --top 25       # show more rows
  python procio_monitor.py --once         # single window result (no continuous refresh)
  python procio_monitor.py --csv out.csv  # also append results to CSV each interval

Notes:
- Counters are cumulative per process since start. We compute deltas between snapshots.
- Some fields may be unavailable on some OSes; we handle gracefully.
- You may need elevated privileges to read all processes' I/O counters.
"""

import argparse
import csv
import os
import sys
import time
import shutil
from datetime import datetime

try:
    import psutil
except ImportError:
    print("This tool requires the 'psutil' package. Install with: pip install psutil", file=sys.stderr)
    sys.exit(1)


def fmt_bytes(n):
    try:
        n = float(n)
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if n >= 100:
        return f"{n:,.0f} {units[i]}"
    elif n >= 10:
        return f"{n:,.1f} {units[i]}"
    else:
        return f"{n:,.2f} {units[i]}"


def get_snapshot():
    snap = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "username", "create_time"]):
        pid = proc.info["pid"]
        try:
            io = proc.io_counters()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if not io:
            continue
        snap[pid] = {
            "name": proc.info.get("name") or "",
            "cmdline": " ".join(proc.info.get("cmdline") or [])[:300],
            "username": proc.info.get("username") or "",
            "create_time": proc.info.get("create_time") or 0.0,
            "read_bytes": getattr(io, "read_bytes", 0),
            "write_bytes": getattr(io, "write_bytes", 0),
            "read_count": getattr(io, "read_count", 0),
            "write_count": getattr(io, "write_count", 0),
        }
    return snap


def compute_deltas(prev, curr, interval):
    rows = []
    for pid, c in curr.items():
        p = prev.get(pid)
        if not p:
            # No previous data (new process) — skip this interval to avoid inflated rates
            continue
        read_b = max(0, c["read_bytes"] - p["read_bytes"])
        write_b = max(0, c["write_bytes"] - p["write_bytes"])
        read_ops = max(0, c["read_count"] - p["read_count"])
        write_ops = max(0, c["write_count"] - p["write_count"])
        rows.append({
            "pid": pid,
            "name": c["name"],
            "username": c["username"],
            "cmdline": c["cmdline"],
            "age_s": max(0.0, (time.time() - c["create_time"]) if c["create_time"] else 0.0),
            "read_bps": read_b / interval if interval > 0 else 0.0,
            "write_bps": write_b / interval if interval > 0 else 0.0,
            "total_bps": (read_b + write_b) / interval if interval > 0 else 0.0,
            "read_ops": read_ops,
            "write_ops": write_ops,
        })
    return rows


def print_table(rows, sort_key, top, interval, header=True):
    # Terminal width for nice truncation of cmd/name
    width = shutil.get_terminal_size((120, 25)).columns
    cols = [
        ("PID", 7),
        ("USER", 10),
        ("NAME", 18),
        ("READ/s", 11),
        ("WRITE/s", 11),
        ("TOTAL/s", 11),
        ("rOPS", 6),
        ("wOPS", 6),
        ("CMD", max(10, width - (7+10+18+11+11+11+6+6+7)))  # slack + spacing between columns
    ]

    # Sort
    rows_sorted = sorted(rows, key=lambda r: r.get(sort_key, 0.0), reverse=True)[:top]

    # Header
    if header:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] interval={interval:.2f}s  sort={sort_key}  top={top}")
        print("-" * width)
        # Print column headers
        line = []
        for title, colw in cols:
            line.append(title.ljust(colw))
        print(" ".join(line))
        print("-" * width)

    # Rows
    for r in rows_sorted:
        fields = [
            str(r["pid"]).ljust(cols[0][1]),
            r["username"][:cols[1][1]-1].ljust(cols[1][1]),
            r["name"][:cols[2][1]-1].ljust(cols[2][1]),
            fmt_bytes(r["read_bps"]).rjust(cols[3][1]),
            fmt_bytes(r["write_bps"]).rjust(cols[4][1]),
            fmt_bytes(r["total_bps"]).rjust(cols[5][1]),
            str(r["read_ops"]).rjust(cols[6][1]),
            str(r["write_ops"]).rjust(cols[7][1]),
            r["cmdline"][:cols[8][1]-1].ljust(cols[8][1]),
        ]
        print(" ".join(fields))


def maybe_write_csv(rows, csv_path, sort_key, top, interval):
    if not csv_path:
        return
    # Append mode; write header if file doesn't exist
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "interval_s", "sort", "pid", "user", "name",
                        "read_Bps", "write_Bps", "total_Bps", "read_ops", "write_ops", "cmdline"])
        ts = datetime.now().isoformat()
        for r in sorted(rows, key=lambda r: r.get(sort_key, 0.0), reverse=True)[:top]:
            w.writerow([ts, interval, sort_key, r["pid"], r["username"], r["name"],
                        f"{r['read_bps']:.6f}", f"{r['write_bps']:.6f}", f"{r['total_bps']:.6f}",
                        r["read_ops"], r["write_ops"], r["cmdline"]])


def main():
    ap = argparse.ArgumentParser(description="Continuously list processes by disk I/O throughput (bytes/sec).")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    ap.add_argument("--sort", choices=["read_bps", "write_bps", "total_bps"], default="write_bps",
                    help="Sort key (default: write_bps)")
    ap.add_argument("--top", type=int, default=15, help="Number of rows to display (default: 15)")
    ap.add_argument("--once", action="store_true", help="Run a single interval and exit")
    ap.add_argument("--csv", type=str, default=None, help="Append results to CSV path each interval")
    args = ap.parse_args()

    try:
        prev = get_snapshot()
        time.sleep(args.interval)
        while True:
            curr = get_snapshot()
            rows = compute_deltas(prev, curr, args.interval)

            # Clear screen and print table
            if not args.once:
                # ANSI clear screen and home
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()

            print_table(rows, args.sort, args.top, args.interval, header=True)
            maybe_write_csv(rows, args.csv, args.sort, args.top, args.interval)

            if args.once:
                break

            prev = curr
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
