#date: 2024-11-15T17:10:17Z
#url: https://api.github.com/gists/c88cdcff57437f341fd3b68c989d62ed
#owner: https://api.github.com/users/azambrano-lyft

from datetime import datetime, timedelta
from expressdrive.utils import time_utils
from app.scripts.compute_daily_driver_mileage import compute_mileage

import pytz


def parse_dates_for_calculation(
    date_str: str | None,
) -> tuple[datetime, datetime]:
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format; defaulting to today's date")
            target_date = time_utils.utc_now().date()
    else:
        target_date = time_utils.utc_now().date()

    # Localize the date to UTC timezone at midnight
    localized_start_datetime = pytz.UTC.localize(
        datetime.combine(target_date, datetime.min.time())
    )
    localized_end_datetime = localized_start_datetime + timedelta(days=1)

    return localized_start_datetime, localized_end_datetime


def compute_mileage_to_simulate_processing(
    user_id: int, start_datetime: datetime, end_datetime: datetime
) -> int:
    start_time_ms = time_utils.convert_time_to_epoch_ms(dt=start_datetime)
    end_time_ms = time_utils.convert_time_to_epoch_ms(dt=end_datetime)

    driver_mileage = compute_mileage(
        user_id=user_id,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )

    if driver_mileage is None:
        return 0

    print("Computed P0 mileage:")
    print(driver_mileage)
    return driver_mileage.p0_mileage_m



def example_usage():
    user_id = 1845522317731789792
    date_str = "2024-10-21"
    start_datetime, end_datetime = parse_dates_for_calculation(date_str)

    mileage = compute_mileage_to_simulate_processing(user_id, start_datetime, end_datetime)
    if mileage > 0:
        print(f"Computed mileage for user {user_id} from {start_datetime} to {end_datetime}: {mileage}")
    else:
        print(f"Could not compute mileage for user {user_id} from {start_datetime} to {end_datetime}")


example_usage()
