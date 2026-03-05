#date: 2026-03-05T18:41:56Z
#url: https://api.github.com/gists/7221fcf90e6f269edb9c6aadfa95e047
#owner: https://api.github.com/users/christophertubbs

"""
Prototype for getting a snapshot for currently running processes

Contains functions that will investigate all active processes and yield information such as the exact command that
launched them and what resources they are actively using
"""
import typing
import collections.abc as generic
import time
import traceback
import sys
import re
import enum
import dataclasses
import json

from datetime import datetime
from datetime import timedelta

import psutil


class ByteFactor(enum.IntEnum):
    """
    Defines an enumeration for representing different byte size factors.

    This class provides a set of commonly used byte size factors, including bytes,
    kilobytes, megabytes, and gigabytes. Each factor is represented as an integer
    power of 1000 corresponding to its scale. It can be useful in cases where
    you need to standardize the representation of byte sizes or convert between
    different units of data sizes.

    :ivar BYTES: Represents a single byte (1000^0).
    :type BYTES: int
    :ivar KILOBYTES: Represents one kilobyte (1000^1).
    :type KILOBYTES: int
    :ivar MEGABYTES: Represents one megabyte (1000^2).
    :type MEGABYTES: int
    :ivar GIGABYTES: Represents one gigabyte (1000^3).
    :type GIGABYTES: int
    """
    BYTES = 1000 ** 0
    KILOBYTES = 1000 ** 1
    MEGABYTES = 1000 ** 2
    GIGABYTES = 1000 ** 3


class TimeFactor(enum.IntEnum):
    """
    Represents time conversion factors relative to seconds.

    This enumeration defines constants for various time units
    and the number of seconds each unit corresponds to. It is
    useful for converting between different time scales efficiently
    by providing pre-defined conversion factors.

    :ivar SECONDS: Represents one second (base unit).
    :type SECONDS: int
    :ivar MINUTES: Represents one minute, equivalent to 60 seconds.
    :type MINUTES: int
    :ivar HOURS: Represents one hour, equivalent to 3,600 seconds.
    :type HOURS: int
    :ivar DAYS: Represents one day, equivalent to 86,400 seconds.
    :type DAYS: int
    :ivar WEEKS: Represents one week, equivalent to 604,800 seconds.
    :type WEEKS: int
    """
    SECONDS = 1
    MINUTES = 60
    HOURS = 60 * 60
    DAYS = 60 * 60 * 24
    WEEKS = 60 * 60 * 24 * 7


def format_delta(delta: timedelta | int | float = None) -> str | None:
    """
    Formats a time duration into an ISO 8601 duration string. Acceptable input types
    include `timedelta`, `int`, or `float`. If the provided duration is an integer
    or float, it is converted into a `timedelta` assuming it represents seconds. The
    generated string adheres to the format `PnWnDTnHnMnS`.

    :param delta: Time duration to format. Can be of type `timedelta`, `int`, or
                  `float`. If `None`, the function returns `None`.
    :return: A string representing the ISO 8601 formatted duration, or `None` if
             `delta` is `None`.
    :rtype: str | None
    """
    if delta is None:
        return None

    if isinstance(delta, (int, float)):
        delta = timedelta(seconds=delta)

    seconds: float = delta.total_seconds()
    weeks, seconds = divmod(seconds, TimeFactor.WEEKS)
    days, seconds = divmod(seconds, TimeFactor.DAYS)
    hours, seconds = divmod(seconds, TimeFactor.HOURS)
    minutes, seconds = divmod(seconds, TimeFactor.MINUTES)

    description: str = "P"

    if weeks:
        description += f"{int(weeks)}W"

    if days:
        description += f"{int(days)}D"

    description += "T"

    if hours:
        description += f"{int(hours)}H"

    if minutes:
        description += f"{int(minutes)}M"

    if seconds:
        description += f"{seconds:.2f}S"

    return description


def format_bytes(amount: int) -> str:
    """
    Formats a given amount of bytes into a human-readable string representation with size
    units such as B, KB, MB, or GB. The conversion is done based on commonly used byte
    multipliers, and the result will contain appropriate unit labels.

    :param amount: The size in bytes to be formatted into a human-readable string. Must
        be a non-negative integer.
    :return: A string representing the formatted size with units (e.g., "10.50KB",
        "35.00MB").
    """
    if amount / ByteFactor.BYTES < 1000.0:
        return f"{amount / ByteFactor.BYTES}B"
    if amount / ByteFactor.KILOBYTES < 1000.0:
        return f"{amount / ByteFactor.KILOBYTES:.2f}KB"
    if amount / ByteFactor.MEGABYTES < 1000.0:
        return f"{amount / ByteFactor.MEGABYTES:.2f}MB"
    return f"{amount / ByteFactor.GIGABYTES:.2f}GB"


@dataclasses.dataclass
class Process:
    """
    Representation of a process with hierarchical structure, resource usage, and associated metadata.

    This class encapsulates information about a process, including its metadata, resource consumption,
    and its relationship with child processes. It provides methods to compute and represent the
    process data in various formats and supports recursive memory aggregation for a hierarchical
    process tree.

    :ivar pid: Process ID.
    :type pid: int
    :ivar ppid: Parent Process ID.
    :type ppid: int
    :ivar name: Process name or executable.
    :type name: str
    :ivar elapsed: Time elapsed since the process started, in a human-readable format.
        This attribute may be None if unavailable.
    :type elapsed: str | None
    :ivar created_at: Timestamp of when the process was created. This attribute may be None if unavailable.
    :type created_at: str | None
    :ivar owner: Username of the process owner. This attribute may be None if unavailable.
    :type owner: str | None
    :ivar command: The command used to start the process. This attribute may be None if unavailable.
    :type command: str | None
    :ivar cpu_percent: Percentage of CPU used by the process. This attribute may be None if unavailable.
    :type cpu_percent: float | None
    :ivar rss: Resident Set Size (RSS) memory used by the process in bytes or other numeric formats.
    :type rss: int | float
    :ivar children: List of child processes belonging to this process, represented as instances of
        `Process`. By default, this attribute initializes as an empty list.
    :type children: list[Process]
    """
    pid: int
    ppid: int
    name: str
    elapsed: str | None
    created_at: str | None
    owner: str | None
    command: str | None
    cpu_percent: float | None
    rss: int | float
    children: list["Process"] = dataclasses.field(default_factory=list)

    @property
    def total_memory(self) -> float | int:
        total: int | float = sum([self.rss, *map(lambda child: child.total_memory, self.children)])
        return total

    @property
    def memory(self) -> str:
        return format_bytes(self.total_memory)

    def to_dict(self):
        _data: dict[str, typing.Any] = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("children", "rss")
        }
        _data["total_bytes"] = self.total_memory
        _data["memory"] = self.memory
        _data["children"] = [child.to_dict() for child in self.children]
        return _data


def snapshot_processes(
    filter_expression: re.Pattern | str | bytes | None = None
) -> generic.Sequence[Process]:
    """
    Captures a snapshot of the currently running processes on the system, optionally filtered by a
    regular expression pattern or string. Each process is summarized with relevant details such as
    process ID, name, command, owner, resource usage, and creation time.

    This function utilizes the `psutil` library to iterate over active system processes and collect
    information. It supports filtering processes based on their command string, using a regular
    expression or substring match. If a filter expression is specified, only matching processes are
    included in the resulting snapshot.

    :param filter_expression: An optional regular expression pattern, string, or bytes object to filter
        processes based on their command string. If a bytes object is provided, it will be decoded.
        If omitted or set to None, no filtering is performed.
    :return: A sequence containing instances of the `Process` class, representing processes running
        on the system at the time of the snapshot.
    :rtype: generic.Sequence[Process]
    """
    current_processes: list[Process] = []
    snapshot_time: float = time.time()

    if isinstance(filter_expression, bytes):
        filter_expression = filter_expression.decode()

    if isinstance(filter_expression, str):
        filter_expression = re.compile(filter_expression)

    for process in psutil.process_iter():
        try:
            command_line_arguments: list[str] = process.cmdline()
            if command_line_arguments:
                command: str = ' '.join(command_line_arguments)
            else:
                command: str = "<Unknown>"

            if isinstance(filter_expression, re.Pattern):
                match: re.Match[str] = filter_expression.search(command)

                if match is None:
                    continue

            process_data: dict[str, typing.Any] = {
                "pid": process.pid,
                "ppid": process.ppid(),
                "name": process.name(),
                "owner": process.username(),
                "elapsed": None,
                "command": command,
                "cpu_percent": process.cpu_percent(interval=0.0),
                "rss": -1,
                "created_at": None
            }

            creation_time: float = process.create_time()
            if creation_time is not None:
                process_data['elapsed'] = format_delta(snapshot_time-creation_time)
                process_data['created_at'] = datetime.fromtimestamp(creation_time).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")

            memory_info: psutil.pmem = process.memory_info()
            process_data['rss'] = memory_info.rss

            process_instance: Process = Process(**process_data)
            current_processes.append(process_instance)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except BaseException as e:
            traceback.print_exc()
            continue

    return current_processes


def build_process_tree(
    processes: generic.Sequence[Process]
) -> generic.Sequence[Process]:
    """
    Builds a hierarchical tree structure of processes based on parent-child
    relationships derived from the provided sequence of process objects. Each
    process will either be added as a root process or attached to its parent’s
    child list depending on whether its parent process exists in the sequence.

    :param processes: A sequence of `Process` objects used to construct the
        process tree. Each `Process` instance must contain valid `pid` and
        `ppid` attributes, where `pid` is the process identifier, and `ppid`
        is the parent process identifier.
    :return: A sequence of `Process` objects representing the root processes
        of the constructed tree. The roots are sorted alphabetically by their
        `name` attribute.
    :rtype: generic.Sequence[Process]
    """
    process_by_pid: dict[int, Process] = {
        process.pid: process
        for process in processes
    }

    root_processes: list[Process] = []

    for process in processes:
        parent_process_id: int = process.ppid

        parent_process: Process | None = process_by_pid.get(parent_process_id)

        if parent_process is None:
            root_processes.append(process)
        else:
            parent_process.children.append(process)

    return sorted(root_processes, key=lambda entry: entry.name)


def main() -> int:
    """
    Executes the main logic of the program, which captures currently running system
    processes, builds a hierarchical process tree, and outputs the resulting
    data in JSON format. Handles anticipated interruptions and generic errors by
    displaying relevant messages and adjusting the program's exit code accordingly.

    :return: An integer exit code indicating the success (0) or failure (1) of
        the execution.
    :rtype: int
    """
    exit_code: int = 0
    try:
        processes: generic.Sequence[Process] = snapshot_processes(
            sys.argv[1] if len(sys.argv) > 1 else None
        )
        process_tree: generic.Sequence[Process] = build_process_tree(processes=processes)
        process_data: list[dict] = [process.to_dict() for process in process_tree]
        print(json.dumps(process_data, indent=4))
    except KeyboardInterrupt:
        print("Prototype interrupted")
    except BaseException:
        print("Error encountered", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
