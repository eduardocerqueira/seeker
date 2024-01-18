#date: 2024-01-18T16:51:42Z
#url: https://api.github.com/gists/65967225b80de5fc4ba57bb6d84772cd
#owner: https://api.github.com/users/seidtgeist

import pytz
from datetime import datetime, timedelta
import csv

timezones = [
    "Europe/Berlin",
    "Australia/Sydney",
]

timezones = pytz.common_timezones
# timezones = pytz.all_timezones

# start at start of day
start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

# end in 21 years from now
end = start.replace(year=start.year + 21)

# Open the file in write mode
with open("output.csv", mode="w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(
        [
            "tz_name",
            "utc_offset",
            "utc_transition_time",
            "local_transition_time",
            "transition_type",
        ]
    )

    for tz_name in timezones:
        timezone = pytz.timezone(tz_name)

        if not hasattr(timezone, "_utc_transition_times"):
            print("Skipping", tz_name)
            continue

        for utc_transition_time, transition_info in zip(
            timezone._utc_transition_times, timezone._transition_info
        ):
            if utc_transition_time >= start and utc_transition_time < end:
                dst_offset = transition_info[1]
                transition_type = (
                    "dst_end" if dst_offset == timedelta(0) else "dst_start"
                )

                try:
                    # determine the offset at tz_name from UTC so we can calculate the local date
                    utc_offset = timezone.utcoffset(utc_transition_time)
                except pytz.AmbiguousTimeError:
                    utc_offset = timezone.utcoffset(
                        utc_transition_time - timedelta(days=1)
                    )
                    print(
                        f"AmbiguousTimeError {tz_name} subtracting days=1 from {utc_transition_time}"
                    )
                    pass
                except pytz.NonExistentTimeError:
                    utc_offset = timezone.utcoffset(
                        utc_transition_time - timedelta(days=1)
                    )
                    print(
                        f"NonExistentTimeError {tz_name} subtracting days=1 from {utc_transition_time}"
                    )
                    pass

                # determinte the local datetime of the transition
                local_datetime = utc_transition_time + utc_offset

                csv_writer.writerow(
                    [
                        tz_name,
                        utc_offset,
                        utc_transition_time,
                        local_datetime,
                        transition_type,
                    ]
                )