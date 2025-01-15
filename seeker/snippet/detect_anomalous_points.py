#date: 2025-01-15T17:10:40Z
#url: https://api.github.com/gists/12be4ed1d3b2557f4cb05d0d947fa4cc
#owner: https://api.github.com/users/tigattack

import json
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import requests
import typer
from geopy import distance  # type: ignore
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

logging.basicConfig(level=logging.INFO)

POINTS_PER_PAGE = 1000


class OutputFormat(str, Enum):
    JSON = "JSON"
    TEXT = "plaintext"


class UnitType(str, Enum):
    METRIC = "metric"
    IMPERIAL = "imperial"


class DistanceUnit(str, Enum):
    METERS = "meters"
    KILOMETERS = "kilometers"
    FEET = "feet"
    MILES = "miles"


class SpeedUnitShort(str, Enum):
    METRIC = "km/h"
    IMPERIAL = "MPH"

    @staticmethod
    def from_type(unit_type: UnitType) -> "SpeedUnitShort":
        if unit_type == UnitType.METRIC:
            return SpeedUnitShort.METRIC
        elif unit_type == UnitType.IMPERIAL:
            return SpeedUnitShort.IMPERIAL


@dataclass
class DawarichTrackedMonth:
    year: int
    months: list[str]


@dataclass
class DawarichPoint:
    id: int
    latitude: float
    longitude: float
    timestamp: int
    battery_status: float | None = None
    ping: float | None = None
    battery: float | None = None
    tracker_id: str | None = None
    topic: str | None = None
    altitude: float | None = None
    velocity: float | None = None
    trigger: str | None = None
    bssid: str | None = None
    ssid: str | None = None
    connection: str | None = None
    vertical_accuracy: float | None = None
    accuracy: float | None = None
    mode: int | None = None
    inrids: list[str] | None = None
    in_regions: list[str] | None = None
    raw_data: str | None = None
    import_id: str | None = None
    city: str | None = None
    country: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    user_id: int | None = None
    geodata: str | None = None
    visit_id: str | None = None
    reverse_geocoded_at: str | None = None


@dataclass
class AnomalousPoint:
    start_point: DawarichPoint
    end_point: DawarichPoint
    speed: float
    distance_difference: float
    time_difference: timedelta


@dataclass
class DistanceUnitChoices:
    primary: DistanceUnit
    fallback: DistanceUnit


app = typer.Typer()


def get_distance_unit_choices(units: UnitType) -> DistanceUnitChoices:
    if units == UnitType.METRIC:
        return DistanceUnitChoices(
            primary=DistanceUnit.KILOMETERS, fallback=DistanceUnit.METERS
        )
    elif units == UnitType.IMPERIAL:
        return DistanceUnitChoices(
            primary=DistanceUnit.MILES, fallback=DistanceUnit.FEET
        )


def convert_distance(
    value: float, from_unit: DistanceUnit, to_unit: DistanceUnit
) -> float:
    # Convert to meters first
    meters = value
    if from_unit == DistanceUnit.KILOMETERS:
        meters = value * 1000
    elif from_unit == DistanceUnit.FEET:
        meters = value * 0.3048
    elif from_unit == DistanceUnit.MILES:
        meters = value * 1609.344

    # Convert from meters to target unit
    if to_unit == DistanceUnit.METERS:
        return meters
    elif to_unit == DistanceUnit.KILOMETERS:
        return meters / 1000
    elif to_unit == DistanceUnit.FEET:
        return meters / 0.3048
    elif to_unit == DistanceUnit.MILES:
        return meters / 1609.344

    return meters


def format_speed(speed: float, units: UnitType) -> str:
    """Round the given speed to 2 decimal places and append the relevant speed unit name for the given `units`."""
    if speed == float("inf"):
        return "Infinite"

    speed_round = str(round(speed, 2))
    speed_unit = SpeedUnitShort.from_type(units).value
    return " ".join([speed_round, speed_unit])


def format_distance(distance: float, units: UnitType) -> str:
    """
    Convert the distance to the most relevant unit, round to 2 decimal places, and append the relevant distance unit name.
    """
    unit_choices = get_distance_unit_choices(units)
    converted_distance = convert_distance(
        distance, DistanceUnit.METERS, unit_choices.primary
    )
    if converted_distance >= 1:
        distance_round = str(round(converted_distance, 2))
        distance_unit = unit_choices.primary.value
    else:
        converted_distance = convert_distance(
            distance, DistanceUnit.METERS, unit_choices.fallback
        )
        distance_round = str(round(converted_distance, 2))
        distance_unit = unit_choices.fallback.value

    return " ".join([distance_round, distance_unit])


def get_tracked_months(base_url: str, api_key: str) -> list[DawarichTrackedMonth]:
    """Retrieve tracked months from the API."""
    url = f"{base_url}/points/tracked_months"
    try:
        response = requests.get(url, params={"api_key": api_key})
        response.raise_for_status()
        tracked_months_raw = response.json()
    except requests.RequestException as e:
        logging.error(f"HTTP error occurred: {e}")
        return []
    except ValueError as e:
        logging.error(f"JSON decoding failed: {e}")
        return []

    tracked_months = [DawarichTrackedMonth(**month) for month in tracked_months_raw]

    # Sort each month list within the years
    for month_obj in tracked_months:
        # Convert month abbreviations to numbers for sorting
        month_nums = [(datetime.strptime(m, "%b"), m) for m in month_obj.months]
        month_nums.sort(key=lambda x: x[0].month)
        month_obj.months = [m[1] for m in month_nums]

    tracked_months.sort(key=lambda x: x.year)
    return tracked_months


def get_points(
    base_url: str,
    api_key: str,
    start_at: str,
    end_at: str,
    per_page: int = POINTS_PER_PAGE,
) -> list[DawarichPoint]:
    """Retrieve geopoints from the API."""
    url = f"{base_url}/points"
    params: dict[str, str | int] = {
        "api_key": api_key,
        "start_at": start_at,
        "end_at": end_at,
        "per_page": per_page,
    }
    params["page"] = 1
    all_points: list[DawarichPoint] = []

    while True:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            points_raw = response.json()
        except requests.RequestException as e:
            logging.error(f"HTTP error occurred: {e}")
            break
        except ValueError as e:
            logging.error(f"JSON decoding failed: {e}")
            break

        if not points_raw:
            break

        all_points.extend(DawarichPoint(**point) for point in points_raw)
        params["page"] += 1

    return all_points


def calculate_distance(
    point1: DawarichPoint, point2: DawarichPoint
) -> distance.distance:
    """Calculate the distance between two points in the specified unit."""
    coords_1 = (point1.latitude, point1.longitude)
    coords_2 = (point2.latitude, point2.longitude)
    return distance.distance(coords_1, coords_2)


def calculate_speed(
    point1: DawarichPoint,
    point2: DawarichPoint,
    distance_between: float,
    units: UnitType,
) -> float:
    """Calculate the speed between two points."""
    time_diff_seconds = abs(point2.timestamp - point1.timestamp)
    if time_diff_seconds == 0:
        return float("inf")  # Avoid division by zero

    # Calculate time difference in hours
    time_diff_hours = time_diff_seconds / 3600

    # Calculate speed in the specified unit
    distance_converted = convert_distance(
        distance_between,
        DistanceUnit.METERS,
        get_distance_unit_choices(units).primary,
    )

    return distance_converted / time_diff_hours


def identify_anomalous_movements(
    points: list[DawarichPoint],
    speed_threshold: float,
    distance_threshold: float,
    units: UnitType,
) -> list[AnomalousPoint]:
    """Identify movements exceeding the defined thresholds."""
    anomalies: list[AnomalousPoint] = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i + 1]

        distance = calculate_distance(point1, point2)
        distance_meters = distance.meters  # type: ignore

        # Convert distance_threshold to meters
        converted_distance_threshold = convert_distance(
            distance_threshold,
            get_distance_unit_choices(units).primary,
            DistanceUnit.METERS,
        )

        speed = calculate_speed(
            point1,
            point2,
            distance_meters,  # type: ignore
            units,
        )

        # Check if the speed exceeds speed threshold or distance exceeds distance
        # threshold if speed is incalculable between points (e.g. same timestamp)
        if (speed != float("inf") and speed >= speed_threshold) or (
            speed == float("inf") and distance_meters >= converted_distance_threshold
        ):
            p1_time = datetime.fromtimestamp(point1.timestamp)
            p2_time = datetime.fromtimestamp(point2.timestamp)
            time_diff = p2_time - p1_time
            anomalies.append(
                AnomalousPoint(
                    start_point=point1,
                    end_point=point2,
                    speed=speed,
                    distance_difference=distance_meters,  # type: ignore
                    time_difference=time_diff,
                )
            )
    return anomalies


def anomalies_to_json(anomalies: list[AnomalousPoint]) -> str:
    """Convert anomalies to a JSON string, converting distance from kilometers to meters."""
    anomalies_updated: list[dict[str, Any]] = []
    for anomaly in anomalies:
        if anomaly.speed == float("inf"):
            speed = -1
        else:
            speed = convert_distance(
                anomaly.speed, DistanceUnit.KILOMETERS, DistanceUnit.METERS
            )

        anom = {
            "start_point": anomaly.start_point.__dict__,
            "end_point": anomaly.end_point.__dict__,
            "speed": speed,
        }
        anomalies_updated.append(anom)
    return json.dumps(anomalies_updated, default=str, indent=4)


def anomalies_to_plaintext(anomalies: list[AnomalousPoint], units: UnitType) -> str:
    """Convert anomalies to a formatted, plaintext string."""
    out_data = ""
    for anomaly in anomalies:
        speed = format_speed(anomaly.speed, units)
        distance = format_distance(anomaly.distance_difference, units)
        out_data += f"""Start point:
    ID: {anomaly.start_point.id}
    Latitude: {anomaly.start_point.latitude}
    Longitude: {anomaly.start_point.longitude}
    Timestamp: {datetime.fromtimestamp(anomaly.start_point.timestamp)}
    Location: {anomaly.start_point.city}, {anomaly.start_point.country}
End point:
    ID: {anomaly.end_point.id}
    Latitude: {anomaly.end_point.latitude}
    Longitude: {anomaly.end_point.longitude}
    Timestamp: {datetime.fromtimestamp(anomaly.end_point.timestamp)}
    Location: {anomaly.end_point.city}, {anomaly.end_point.country}
Difference:
    Distance: {distance}
    Time: {anomaly.time_difference}
Movement speed: {speed}\n
"""
    return out_data


def dump_anomalies(
    anomalies: list[AnomalousPoint],
    units: UnitType,
    output_file: str | None,
    output_format: OutputFormat,
) -> None:
    """Handle output formatting and writing anomalies to a file or stdout."""
    if output_format == OutputFormat.JSON:
        out_data = anomalies_to_json(anomalies)
    elif output_format == OutputFormat.TEXT:
        out_data = anomalies_to_plaintext(anomalies, units)

    if output_file:
        with open(output_file, "w") as f:
            f.write(out_data)
    else:
        print("\n\n")
        typer.echo(out_data)


def fetch_points_for_month(
    base_url: str, api_key: str, year: int, month: int
) -> list[DawarichPoint]:
    """Fetch points for a specific year and month."""
    start_at = f"{year}-{month:02d}-01T00:00:00Z"
    _, end_day = monthrange(year, month)
    end_at = f"{year}-{month:02d}-{end_day}T23:59:59Z"

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        progress.add_task(
            f"[blue]Fetching points for {month}/{year}...",
            total=None,
        )
        points = get_points(base_url, api_key, start_at, end_at)
        points.sort(key=lambda x: x.timestamp)
    return points


def fetch_points(base_url: str, api_key: str, year: int | None) -> list[DawarichPoint]:
    """Fetch all points from all tracked months."""
    all_points: list[DawarichPoint] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        progress.add_task("[blue]Fetching tracked months...")
        tracked_months = get_tracked_months(base_url, api_key)
        if year:
            tracked_months = [month for month in tracked_months if month.year == year]
    total_months = sum(len(tracked_month.months) for tracked_month in tracked_months)
    rprint(f"[cyan]Found [bold]{total_months}[/bold] tracked months")

    if not tracked_months:
        rprint("[red]No tracked months found.")
        raise typer.Exit(code=0)

    for tracked in tracked_months:
        year = tracked.year
        for month in tracked.months:
            month_number = datetime.strptime(month, "%b").month
            points = fetch_points_for_month(base_url, api_key, year, month_number)
            all_points.extend(points)
    return all_points


@app.command()
def main(
    base_url: str = typer.Option(
        envvar="DAWARICH_BASE_URL", help="Base URL of the Dawarich API"
    ),
    api_key: str = typer.Option(
        envvar="DAWARICH_API_KEY", help="API key for authentication"
    ),
    check_year: int | None = typer.Option(
        None, help="Year to filter points (optional)"
    ),
    check_month: int | None = typer.Option(
        None, help="Month number to filter points (optional)", min=1, max=12
    ),
    units: UnitType = typer.Option(
        UnitType.METRIC,
        help="Unit system for distances and speeds. NOTE: Speed will always be m/s in JSON output.",
    ),
    speed_threshold: float = typer.Option(
        200,
        help="Maximum allowed speed between two consecutive points. Value respects the selected unit system.",
    ),
    distance_threshold: float = typer.Option(
        0,
        help="Minimum distance between two consecutive points. Value is taken as either MPH or km/h depending on the selected unit system.",
    ),
    output_file: str | None = typer.Option(
        None, help="Output file to dump the anomalies"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TEXT, help="Output format for the dump"
    ),
):
    """
    Identify anomalous movements between geopoints in Dawarich.

    Anomalous movements are detected by comparing the speed and distance between two consecutive points against a threshold.

    If the speed exceeds the threshold, or the speed is incalculable (infinite/-1) and the distance between the points is greater than the threshold, the movement is flagged as anomalous.

    Points can be analysed from a specific year, month, or all time.
    """
    if not base_url:
        base_url = typer.prompt("Enter the Base URL of the API")

    if not api_key:
        api_key = typer.prompt("Enter the API key", hide_input=True)

    # Fetch points based on the year and month provided or fetch all points
    if check_year and check_month:
        all_points = fetch_points_for_month(base_url, api_key, check_year, check_month)
    else:
        all_points = fetch_points(base_url, api_key, check_year)

    if not all_points:
        typer.echo("No points found.")
        return

    rprint(f"[cyan]Retrieved [bold]{len(all_points)}[/bold] points")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        progress.add_task("[blue]Identifying anomalies...", total=None)
        anomalies = identify_anomalous_movements(
            all_points, speed_threshold, distance_threshold, units
        )

    if anomalies:
        rprint(f"[red]Found [bold]{len(anomalies)}[/bold] anomalous movements!")
        dump_anomalies(anomalies, units, output_file, output_format)
        if output_file:
            rprint(
                f"[cyan]Anomalies dumped to file {output_file} in {output_format.value} format"
            )
    else:
        rprint("[green]No anomalies detected! 🎉")


if __name__ == "__main__":
    app()
