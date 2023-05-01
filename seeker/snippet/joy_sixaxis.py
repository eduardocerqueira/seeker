#date: 2023-05-01T16:52:24Z
#url: https://api.github.com/gists/68f1a34847932f7d9a40b8b390558189
#owner: https://api.github.com/users/miuele

from typing import NamedTuple

class SixAxisTriggers(NamedTuple):
    l2: float
    r2: float

class SixAxisSticks(NamedTuple):
    l3: tuple[float, float]
    r3: tuple[float, float]

class SixAxisButtons(NamedTuple):
    l1: bool
    r1: bool
    l2: bool
    r2: bool
    l3: bool
    r3: bool
    circle: bool
    square: bool
    triangle: bool
    cross: bool
    left: bool
    right: bool
    up: bool
    down: bool
    start: bool
    select: bool
    ps: bool

class SixAxis(NamedTuple):
    sticks: SixAxisSticks
    triggers: SixAxisTriggers
    buttons: SixAxisButtons

def translate(joy):
    return SixAxis(
        sticks=SixAxisSticks(
            l3=(joy.axes[0], joy.axes[1]),
            r3=(joy.axes[3], joy.axes[4]),
            ),
        triggers=SixAxisTriggers(
            l2=joy.axes[2],
            r2=joy.axes[5],
            ),
        buttons=SixAxisButtons(
            l1=bool(joy.buttons[4]),
            r1=bool(joy.buttons[5]),
            l2=bool(joy.buttons[6]),
            r2=bool(joy.buttons[7]),
            l3=bool(joy.buttons[11]),
            r3=bool(joy.buttons[12]),
            circle=bool(joy.buttons[1]),
            square=bool(joy.buttons[3]),
            triangle=bool(joy.buttons[2]),
            cross=bool(joy.buttons[0]),
            left=bool(joy.buttons[15]),
            right=bool(joy.buttons[16]),
            up=bool(joy.buttons[13]),
            down=bool(joy.buttons[14]),
            select=bool(joy.buttons[8]),
            start=bool(joy.buttons[9]),
            ps=bool(joy.buttons[10]),
            ),
        )
