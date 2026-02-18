#date: 2026-02-18T17:47:35Z
#url: https://api.github.com/gists/7199b226fb8e13c4d35b5970dd49a71e
#owner: https://api.github.com/users/annawoodard

#!/usr/bin/env python3
"""Project quota runout trajectory from slash-status style text."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def parse_remaining(status_text: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", status_text)
    if not m:
        _fail("could not find remaining percentage in status text")
    value = float(m.group(1))
    if not (0.0 <= value <= 100.0):
        _fail(f"remaining percentage out of range: {value}")
    return value


def extract_reset_text(status_text: str) -> str:
    m = re.search(r"reset(?:s)?(?:\s+at)?\s+(.+)$", status_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return status_text.strip()


def parse_reset(reset_text: str, now: dt.datetime) -> dt.datetime:
    text = reset_text.strip().lower()
    # Normalize common status punctuation/connector noise:
    # "14:01 on 23 Feb)" -> "14:01 on 23 feb"
    text = re.sub(r"[()\[\],]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    pat_time_day_month = re.search(
        r"(?P<time>\d{1,2}:\d{2})\s+(?:on\s+)?(?P<day>\d{1,2})\s+(?P<month>[a-z]{3,9})(?:\s+(?P<year>\d{2,4}))?",
        text,
    )
    pat_day_month_time = re.search(
        r"(?P<day>\d{1,2})\s+(?P<month>[a-z]{3,9})(?:\s+(?P<year>\d{2,4}))?\s+(?:at\s+)?(?P<time>\d{1,2}:\d{2})",
        text,
    )
    pat_time_only = re.search(r"(?P<time>\d{1,2}:\d{2})", text)

    if pat_time_day_month:
        m = pat_time_day_month
    elif pat_day_month_time:
        m = pat_day_month_time
    elif pat_time_only:
        m = pat_time_only
    else:
        _fail(f"could not parse reset time/date from: '{reset_text}'")

    hour_str, minute_str = m.group("time").split(":")
    hour = int(hour_str)
    minute = int(minute_str)
    if hour > 23 or minute > 59:
        _fail(f"invalid reset time: {m.group('time')}")

    day_s = m.groupdict().get("day")
    month_s = m.groupdict().get("month")
    year_s = m.groupdict().get("year")

    if day_s and month_s:
        day = int(day_s)
        month_key = month_s[:3]
        month = MONTHS.get(month_key)
        if month is None:
            _fail(f"invalid month: {month_s}")

        if year_s:
            year = int(year_s)
            if year < 100:
                year += 2000
            candidate = dt.datetime(year, month, day, hour, minute, tzinfo=now.tzinfo)
        else:
            year = now.year
            candidate = dt.datetime(year, month, day, hour, minute, tzinfo=now.tzinfo)
            if candidate <= now:
                candidate = dt.datetime(year + 1, month, day, hour, minute, tzinfo=now.tzinfo)
        return candidate

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + dt.timedelta(days=1)
    return candidate


def fmt_td(delta: dt.timedelta) -> str:
    seconds = int(round(delta.total_seconds()))
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days > 0:
        return f"{sign}{days}d {hours}h {minutes}m"
    return f"{sign}{hours}h {minutes}m"


def parse_now(now_text: str | None) -> dt.datetime:
    if not now_text:
        return dt.datetime.now().astimezone()
    try:
        parsed = dt.datetime.fromisoformat(now_text)
    except ValueError as exc:
        _fail(f"invalid --now datetime '{now_text}': {exc}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate quota trajectory from slash status text. "
            "Example: quota_trajectory.py --status '91%% left resets 14:01 23 feb'"
        )
    )
    parser.add_argument("--status", help="Raw status text (contains percent + reset).")
    parser.add_argument(
        "--remaining",
        type=float,
        help="Remaining quota percent (0-100). Overrides parsing from --status.",
    )
    parser.add_argument(
        "--reset",
        help=(
            "Reset time/date text (e.g. '14:01 23 feb'). "
            "Overrides parsing from --status."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=7.0,
        help="Quota window length in days used for trajectory math (default: 7).",
    )
    parser.add_argument(
        "--now",
        help="Optional ISO datetime for reproducible checks (default: current local time).",
    )
    args = parser.parse_args()

    if args.window_days <= 0:
        _fail("--window-days must be positive")

    if args.remaining is None and not args.status:
        _fail("provide either --status or --remaining")
    if args.reset is None and not args.status:
        _fail("provide either --status or --reset")

    now = parse_now(args.now)

    remaining_pct = args.remaining
    if remaining_pct is None:
        remaining_pct = parse_remaining(args.status)
    if not (0.0 <= remaining_pct <= 100.0):
        _fail("--remaining must be between 0 and 100")

    reset_text = args.reset or extract_reset_text(args.status)
    reset_at = parse_reset(reset_text, now)

    if reset_at <= now:
        _fail("reset time is not in the future")

    window = dt.timedelta(days=args.window_days)
    window_start = reset_at - window
    used_pct = 100.0 - remaining_pct

    elapsed = now - window_start
    remaining_time = reset_at - now

    elapsed_hours = elapsed.total_seconds() / 3600.0
    remaining_hours = remaining_time.total_seconds() / 3600.0
    window_hours = window.total_seconds() / 3600.0

    expected_used_now = max(0.0, min(100.0, (elapsed_hours / window_hours) * 100.0))
    allowed_now_rate = remaining_pct / remaining_hours

    actual_rate = None
    projected_empty_at = None
    verdict = "UNKNOWN"

    if elapsed_hours > 0 and used_pct > 0:
        actual_rate = used_pct / elapsed_hours
        projected_empty_at = now + dt.timedelta(hours=(remaining_pct / actual_rate))
        if projected_empty_at >= reset_at:
            verdict = "ON TRACK"
        else:
            verdict = "WILL RUN OUT EARLY"
    elif used_pct == 0:
        verdict = "ON TRACK"
    else:
        verdict = "WINDOW NOT STARTED"

    print("quota trajectory")
    print(f"- now: {now:%Y-%m-%d %H:%M %Z}")
    print(f"- reset: {reset_at:%Y-%m-%d %H:%M %Z}")
    print(f"- window start ({args.window_days:g}d before reset): {window_start:%Y-%m-%d %H:%M %Z}")
    print(f"- elapsed in window: {fmt_td(elapsed)}")
    print(f"- time to reset: {fmt_td(remaining_time)}")
    print(f"- used: {used_pct:.1f}%")
    print(f"- remaining: {remaining_pct:.1f}%")
    print(f"- expected used by now (linear): {expected_used_now:.1f}%")

    if actual_rate is not None:
        print(f"- actual burn rate: {actual_rate * 24:.1f}%/day")
    else:
        print("- actual burn rate: n/a")
    print(f"- max safe burn rate from now: {allowed_now_rate * 24:.1f}%/day")

    if projected_empty_at is None:
        print("- projected runout: no runout at current burn")
    else:
        delta = projected_empty_at - reset_at
        relation = "after reset" if delta >= dt.timedelta(0) else "before reset"
        print(
            "- projected runout: "
            f"{projected_empty_at:%Y-%m-%d %H:%M %Z} "
            f"({fmt_td(abs(delta))} {relation})"
        )

    print(f"- verdict: {verdict}")


if __name__ == "__main__":
    main()
