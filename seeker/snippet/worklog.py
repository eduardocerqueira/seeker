#date: 2024-08-19T17:10:50Z
#url: https://api.github.com/gists/61b0e6a685ffd9cd09bf978565e7f7f7
#owner: https://api.github.com/users/rednafi

from datetime import datetime, timedelta
from collections.abc import Iterator
from dataclasses import dataclass
import logging

import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass(slots=True)
class WeekLog:
    """Dataclass to store the worklog data for a week."""

    week_number: int
    start_of_week: str
    end_of_week: str
    days_of_week: list[str]


def generate_weeklogs(year: int) -> Iterator[WeekLog]:
    """Generates a worklog for the given year, yielding each week's data as a WeekLog object."""

    first_day_of_year = datetime(year, 1, 1)
    first_monday = first_day_of_year + timedelta(days=(7 - first_day_of_year.weekday()) % 7)

    current_day = first_monday
    week_number = first_monday.isocalendar()[1]
    days_of_week = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

    while current_day.year == year:
        start_of_week = current_day.strftime("%Y-%m-%d")
        end_of_week = (current_day + timedelta(days=4)).strftime("%Y-%m-%d")

        yield WeekLog(
            week_number=week_number, start_of_week=start_of_week, end_of_week=end_of_week, days_of_week=days_of_week
        )

        current_day += timedelta(weeks=1)
        week_number += 1


def export_worklog_to_markdown(weeklogs: Iterator[WeekLog], filename: str) -> None:
    """Exports the generated worklog to a markdown file with the appropriate formatting."""

    logger.info(f"Exporting worklog to {filename}")
    with open(filename, "w") as file:
        # Write the main header
        file.write("# Worklog\n\n")

        # Process each week's data
        for week in weeklogs:
            # Write the week header
            week_num = week.week_number
            start_of_week = week.start_of_week
            end_of_week = week.end_of_week
            days_of_week = week.days_of_week

            file.write(f"## Week {week_num} [{start_of_week} - {end_of_week}]\n\n")

            # Write each day of the week
            for day in days_of_week:
                file.write(f"- **{day}**\n")

            file.write("\n")

    logger.info("Worklog exported successfully!")


if __name__ == "__main__":
    year = 2024
    worklog = generate_weeklogs(year)
    export_worklog_to_markdown(worklog, f"worklog_{year}.md")
