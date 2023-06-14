#date: 2023-06-14T16:55:06Z
#url: https://api.github.com/gists/fdbadb6a3628fca0f8321705c306dd71
#owner: https://api.github.com/users/tigattack

"""Autorestic Discord backup report script"""
import argparse
import json
from os import environ
from sys import exit

import requests

WEBHOOK_URL = "your webhook here"
WEBHOOK_POST_RETRIES = 3

AUTORESTIC_ENV_PARTIALS = [
    'SNAPSHOT_ID',
    'PARENT_SNAPSHOT_ID',
    'FILES_ADDED',
    'FILES_CHANGED',
    'FILES_UNMODIFIED',
    'DIRS_ADDED',
    'DIRS_CHANGED',
    'DIRS_UNMODIFIED',
    'ADDED_SIZE',
    'PROCESSED_FILES',
    'PROCESSED_SIZE',
    'PROCESSED_DURATION',
]


def get_autorestic_env(env_partials: dict) -> dict:
    """Get autorestic data from environment variables"""
    autorestic_data = {}
    for var in env_partials:
        autorestic_data[var] = environ[f"AUTORESTIC_{var}_0"]
    autorestic_data['LOCATION'] = environ['AUTORESTIC_LOCATION']
    return autorestic_data


def convert_human_to_bytes(value: str) -> float:
    """Convert a human-readable bytes string, like 'x.x KiB', 'x.x MiB', 'x.x GiB',
    or 'x.x TiB' to bytes, where x is a digit."""
    # Define conversion factors from K/M/G/TiB to bytes
    conversion_factors = {
        'B': 1,
        'KiB': 1024,
        'MiB': 1024 ** 2,
        'GiB': 1024 ** 3,
        'TiB': 1024 ** 4
    }
    # Split size string into size and unit
    try:
        data_value, data_unit = value.split()
    except ValueError as exc:
        raise ValueError(f"Invalid data size string: {value}") from exc

    # Return data size as bytes
    return float(data_value) * conversion_factors[data_unit]


# https://stackoverflow.com/a/1094933/5209106
def convert_bytes_to_human(num, suffix="B") -> str:
    """Convert bytes to human readable format"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi {suffix}"


def round_float(value: float, precision: int = 2) -> float:
    """Round a float to a given precision"""
    return round(float(value), precision)


def calculate_speed(duration: str, size: str) -> float:
    """
    Calculate transfer speed from a duration string like 'm:s' or 'h:m:s' and
    a size string like 'x.x KiB', 'x.x MiB', 'x.x GiB' or 'x.x TiB' where x is a digit.
    """
    # Convert duration string to seconds
    # https://stackoverflow.com/a/41252517/5209106
    duration_seconds = sum(
        x * int(t) for x, t in zip([1, 60, 3600], reversed(duration.split(":")))
    )

    # Convert size string to bytes
    data_bytes = convert_human_to_bytes(size)

    # Calculate backup processing speed based on processed size and duration
    try:
        return float(data_bytes / duration_seconds)
    except ZeroDivisionError:
        return float(data_bytes)


def make_payload(name: str, status: str, colour: int, fields: list) -> dict:
    """Construct Discord payload"""
    return {
        "username": "Restic Backup",
        "avatar_url": "https://restic.readthedocs.io/en/stable/_static/logo.png",
        "embeds": [
            {
                "title": name,
                "description": f"Result: {status}",
                "color": colour,
                "fields": fields
            }
        ]
    }


def main():
    """Do the ting"""
    # # Parse command line arguments
    parser = argparse.ArgumentParser(description='Send backup report message to Discord webhook')
    parser.add_argument(
        '--status',
        type=int,
        required=True,
        choices=[0, 1], # 0 = success, 1 = fail
        help='Status of the backup'
    )
    args = parser.parse_args()

    # Get autorestic data
    autorestic_data = get_autorestic_env(AUTORESTIC_ENV_PARTIALS)

    # Set backup status friendly string
    backup_status = "Success" if args.status == 0 else "Fail"
    embed_colour = 65280 if args.status == 0 else 16711680

    # Get transfer speed
    speed = calculate_speed(
        autorestic_data['PROCESSED_DURATION'],
        autorestic_data['ADDED_SIZE']
    )

    # Round data values
    processed_size_parts = autorestic_data['PROCESSED_SIZE'].split()
    processed_size_rounded = round_float(float(processed_size_parts[0]))
    processed_size_human = str(processed_size_rounded) + ' ' + processed_size_parts[1]

    added_size_parts = autorestic_data['ADDED_SIZE'].split()
    added_size_rounded = round_float(float(added_size_parts[0]))
    added_size_human = str(added_size_rounded) + ' ' + added_size_parts[1]

    speed_rounded = round_float(speed)
    speed_human = str(convert_bytes_to_human(speed_rounded)) + "/s"

    # Get payload
    payload = make_payload(
        name=autorestic_data['LOCATION'],
        status=backup_status,
        colour=embed_colour,
        fields=[
            {
                "name": "Data Processed",
                "value": processed_size_human,
                "inline": True
            },
            {
                "name": "Data Added",
                "value": added_size_human,
                "inline": True
            },
            {
                "name": "Transfer Speed",
                "value": speed_human,
                "inline": True
            },
            {
                "name": "Backup Duration",
                "value": autorestic_data['PROCESSED_DURATION'],
                "inline": True
            },
            {
                "name": "Files Processed",
                "value": autorestic_data['PROCESSED_FILES'],
                "inline": True
            },
            {
                "name": "Snapshot ID",
                "value": autorestic_data['SNAPSHOT_ID'],
                "inline": True
            }
        ]
    )

    # Send message to Discord webhook
    headers = {'Content-Type': 'application/json'}

    for _ in range(WEBHOOK_POST_RETRIES):
        try:
            response = requests.post(
                WEBHOOK_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            # Check if the request was successful
            if response.status_code in [200, 204]:
                print("Request successful!")
                break  # Exit the loop if successful
            print(f"Request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as exc:
            print(f"Request failed with error: {exc}")
        print("Retrying...")

if __name__ == '__main__':
    main()
