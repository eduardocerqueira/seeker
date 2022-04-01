#date: 2022-04-01T16:51:58Z
#url: https://api.github.com/gists/2057c964ab03cbe7e12c156d0402d018
#owner: https://api.github.com/users/erikson1970

from __future__ import annotations
from rich.panel import Panel
from rich.align import Align
from rich.pretty import Pretty
from rich import box
from rich.traceback import Traceback
from rich.console import RenderableType
import rich.repr

from textual import events

from textual.app import App
from textual.reactive import Reactive
from textual.widget import Widget
from textual.message import Message, MessageTarget

from textual.widgets import Placeholder, Footer, Header, Button, ScrollView


from logging import getLogger

log = getLogger("rich")


@rich.repr.auto
class dataFinal(Message, bubble=True):
    SQL = 0
    TXT = 1
    MD = 2
    CSV = 3

    def __rich_repr__(self) -> rich.repr.Result:
        yield "datatype", ["SQL", "TXT", "MD", "CSV"][self.datatype]
        yield "message:", self.mymessage

    def __init__(
        self, sender: MessageTarget, mymessage: str, datatype: int = 0
    ) -> None:
        self.mymessage = mymessage
        self.datatype = datatype
        super().__init__(sender)


@rich.repr.auto(angular=False)
class Reader(Widget, can_focus=True):
    """Accepts text input from keyboard when has focus. 
    
    Emits content as dataFinal message on hitting [enter]. 
    Definitely a work in progress.
    """
    has_focus: Reactive[bool] = Reactive(False)
    mouse_over = Reactive(False)
    style: Reactive[str] = Reactive("")
    height: Reactive[int | None] = Reactive(None)
    Reader_count: int = 0

    def __init__(
        self,
        *,
        name: str | None = None,
        height: int | None = None,
        showdetail: int = 0,
        datatype: int = dataFinal.SQL,
    ) -> None:
        super().__init__(name=name)
        self.height = height
        self.showdetail = showdetail
        self.name = (
            name
            if name is not None
            else f"{self.__class__.__name__}_{Reader.Reader_count}"
        )
        self.datatype = datatype
        self.NNdata = None
        self.dataHistory = []
        Reader.Reader_count += 1

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "Data:", self.NNdata
        yield "has_focus", self.has_focus, False
        yield "mouse_over", self.mouse_over, False

    def render(self) -> RenderableType:
        return Panel(
            Align.center(
                Pretty(
                    self
                    if self.showdetail > 1
                    else (self.dataHistory if self.showdetail > 0 else self.NNdata)
                ),
                vertical="middle",
            ),
            title=self.name,
            border_style="green" if self.mouse_over else "blue",
            box=box.DOUBLE_EDGE if self.has_focus else box.ROUNDED,
            style=self.style,
            height=self.height,
        )

    async def on_focus(self, event: events.Focus) -> None:
        self.has_focus = True

    async def on_blur(self, event: events.Blur) -> None:
        self.has_focus = False

    async def on_enter(self, event: events.Enter) -> None:
        self.mouse_over = True

    async def on_leave(self, event: events.Leave) -> None:
        self.mouse_over = False

    async def on_key(self, event: events.Key):
        if self.has_focus:
            if event.key == "enter" or event.key == "ctrl+r":
                msg = dataFinal(self, self.NNdata)
                self.dataHistory = [self.NNdata] + self.dataHistory
                await self.emit(dataFinal(self, self.NNdata, self.datatype))
                self.NNdata = None
            elif event.key == "left" or event.key == "ctrl+h":
                self.NNdata = self.NNdata[:-1] if not self.NNdata is None else None
            elif event.key == "ctrl+d":
                self.showdetail = (self.showdetail + 1) % 3
            elif event.key == "up":
                self.dataHistory.append(self.NNdata)
                self.NNdata = self.dataHistory[0]
                self.dataHistory = self.dataHistory[1:] + [self.dataHistory[0]]
            elif event.key.isalnum() or event.key[0] in "!@#$%^&*()[]\{\}-+,.;_<>='\" ":
                if self.NNdata is None:
                    self.NNdata = event.key
                else:
                    self.NNdata = self.NNdata + event.key
            else:
                self.log(
                    f"READER: {self.name} on_key event - unknown key '{event.key}'"
                )
            self.refresh()


if __name__ == "__main__":

    class ReaderApp(App):
        """Demonstrates Character Reading Class"""

        async def on_load(self, event: events.Load) -> None:
            """Bind keys with the app loads (but before entering application mode)"""
            await self.bind("up", "view.toggle('Reader_0')", "Toggle sidebar")
            # await self.bind("q", "quit", "Quit")
            await self.bind("escape", "quit", "Quit")

        async def on_mount(self) -> None:
            self.body = ScrollView(gutter=1)
            """Build layout here."""
            await self.view.dock(Header(), edge="top")
            await self.view.dock(Footer(), edge="bottom")
            readers = (Reader() for _ in range(3))
            await self.view.dock(self.body, edge="right", size=60)
            await self.view.dock(*readers, edge="top")

        async def handle_data_final(self, thismessage: dataFinal) -> None:
            """A message sent by the  reader when change is made."""
            syntax: RenderableType
            try:
                # Construct a Syntax object for the path in the message
                syntax = str(thismessage.mymessage)
                await self.body.update(syntax)
            except Exception:
                # Possibly a binary file
                # For demonstration purposes we will show the traceback
                syntax = Traceback(theme="monokai", width=None, show_locals=True)
                await self.body.update(syntax)

    ReaderApp.run(log="textual.log")
