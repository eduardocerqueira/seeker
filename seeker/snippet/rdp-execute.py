#date: 2024-03-07T18:30:25Z
#url: https://api.github.com/gists/9b9fd8eb083f95e616c450652b1685f6
#owner: https://api.github.com/users/ASkyeye

# Copyright: (c) 2024, Jordan Borean (@jborean93) <jborean93@gmail.com>
# MIT License (see LICENSE or https://opensource.org/licenses/MIT)

"""POC for running exe's over RDP

This is a very rough attempt at trying to run an exe using a headless RDP
connection. It aims to be able to provide an interactive console session as
well as a headless one.

Requires aardwolf - https://github.com/skelsec/aardwolf as a Python dependency.
Also requires the ServerChannel exe from
https://github.com/jborean93/ProcessVirtualChannel to be setup to run at logon
as a scheduled task trigger on the Windows side.

Things that need to be improved before this becomes viable:
    + Figure out a way to log off and not just disconnect a user once done
    + Figure out a way to start the exe without needing the scheduled task
    + Find a proper way to present an actual console on the client side
    + Lots of better logging
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
import termios
import typing as t

from aardwolf.commons.factory import RDPConnectionFactory
from aardwolf.commons.iosettings import RDPIOSettings
from aardwolf.commons.queuedata.constants import VIDEO_FORMAT
from aardwolf.extensions.RDPEDYC.channel import RDPEDYCChannel
from aardwolf.extensions.RDPEDYC.vchannels import VirtualChannelBase

_UNSAFE_C = re.compile('[\\s\t"]')


class ServerChannel(VirtualChannelBase):
    def __init__(
        self,
        event_loop: asyncio.BaseEventLoop,
        res_queue: asyncio.Queue[dict],
        executable: str,
        arguments: str | None,
    ) -> None:
        self.event_loop = event_loop
        self.res_queue = res_queue
        self.executable = executable
        self.arguments = arguments
        super().__init__("ServerChannel")

    async def channel_init(self) -> tuple[bool, Exception | None]:
        return True, None

    async def channel_data_in(self, data: bytes) -> None:
        if data == b"\x00":
            # First connection, we need to tell it what to execute
            process_manifest = json.dumps(
                {
                    "ChannelName": "ServerChannel",
                    "Executable": self.executable,
                    "Arguments": self.arguments,
                }
            ).encode()
            await self.channel_data_out(process_manifest)
            return

        info = json.loads(data)
        self.event_loop.call_soon_threadsafe(self.res_queue.put_nowait, info)

    def __deepcopy__(self, memo: dict) -> ServerChannel:
        return self


class ServerChannelStdin(VirtualChannelBase):
    def __init__(self) -> None:
        super().__init__("ServerChannel-stdin")
        # self._wait_for_input_ready = asyncio.Event()
        self._stdin_task = None

    async def channel_init(self) -> tuple[bool, Exception | None]:
        return True, None

    async def channel_data_in(self, data: bytes) -> None:
        # self._wait_for_input_ready.set()
        self._stdin_task = asyncio.create_task(self._read_stdin())
        return

    @contextlib.contextmanager
    def _raw_mode(self, fileno: int) -> t.Generator:
        old_attrs = termios.tcgetattr(fileno)
        new_attrs = old_attrs[:]
        new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
        try:
            termios.tcsetattr(fileno, termios.TCSADRAIN, new_attrs)
            yield
        finally:
            termios.tcsetattr(fileno, termios.TCSADRAIN, old_attrs)

    async def _read_stdin(self) -> None:
        # await self._wait_for_input_ready.wait()

        # TODO: use custom StreamReaderProtocol
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        proto = asyncio.StreamReaderProtocol(reader)

        with self._raw_mode(sys.stdin.fileno()):
            await loop.connect_read_pipe(lambda: proto, sys.stdin)

            # Line by line
            # while True:
            #     line = await reader.readline()
            #     # print(f"STDIN: sending: '{line}'")
            #     await self.channel_data_out(line)

            # Char by char
            while not reader.at_eof():
                c = await reader.read(1)
                if not c or ord(c) <= 4:
                    break
                await self.channel_data_out(c)


class ServerChannelStdout(VirtualChannelBase):
    def __init__(self) -> None:
        super().__init__("ServerChannel-stdout")

    async def channel_init(self) -> tuple[bool, Exception | None]:
        return True, None

    async def channel_data_in(self, data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


class ServerChannelStderr(VirtualChannelBase):
    def __init__(self) -> None:
        super().__init__("ServerChannel-stderr")

    async def channel_init(self) -> tuple[bool, Exception | None]:
        return True, None

    async def channel_data_in(self, data: bytes) -> None:
        sys.stderr.buffer.write(data)
        sys.stderr.buffer.flush()


def parse_args(
    argv: list[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rdp-execute.py",
        description="Run a process over RDP.",
    )

    parser.add_argument(
        "url",
        action="store",
        nargs=1,
        help="RDP connection URL",
    )

    parser.add_argument(
        "executable",
        action="store",
        nargs=1,
        help="The executable to run",
    )

    parser.add_argument(
        "arguments",
        action="store",
        nargs=argparse.REMAINDER,
        help="The argument for the executable to run",
    )

    return parser.parse_args(argv)


def quote_c_arg(s: str) -> str:
    # https://docs.microsoft.com/en-us/archive/blogs/twistylittlepassagesallalike/everyone-quotes-command-line-arguments-the-wrong-way
    if not s:
        return '""'

    if not _UNSAFE_C.search(s):
        return s

    s = s.replace('"', '\\"')
    s = re.sub(r'(\\+)\\"', r"\1\1\"", s)
    s = re.sub(r"(\\+)$", r"\1\1", s)
    return '"{0}"'.format(s)


async def async_main(
    server_url: str,
    executable: str,
    arguments: list[str],
) -> None:
    # Windows operates on string arguments not a list so we do our best to
    # escape if needed.
    argument_str = " ".join(quote_c_arg(a) for a in arguments)

    res_queue = asyncio.Queue[dict]()
    iosettings = RDPIOSettings()
    iosettings.channels = [RDPEDYCChannel]
    iosettings.vchannels = {
        "ServerChannel": ServerChannel(asyncio.get_event_loop(), res_queue, executable, argument_str),
        "ServerChannel-stdin": ServerChannelStdin(),
        "ServerChannel-stdout": ServerChannelStdout(),
        "ServerChannel-stderr": ServerChannelStderr(),
    }
    iosettings.video_out_format = VIDEO_FORMAT.RAW
    iosettings.clipboard_use_pyperclip = False

    factory = RDPConnectionFactory.from_url(server_url, iosettings)

    async with factory.create_connection_newtarget(factory.target.hostname, iosettings) as connection:
        _, err = await connection.connect()
        if err:
            raise err

        # See if there is a way to just discard this as it comes in
        # while True:
        #     await connection.ext_out_queue.get()

        # First message is the process info like pid/tid
        info = await res_queue.get()
        # print(f"ProcessInfo: {info}")
        if err_msg := info.get("ErrorMessage", None):
            raise Exception(err_msg)

        # Next message is the rc signalling the process has ended.
        res = await res_queue.get()
        # print(f"ProcessResult: {res}")

    sys.exit(res["ReturnCode"])


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    asyncio.run(async_main(args.url[0], args.executable[0], args.arguments))
