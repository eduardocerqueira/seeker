#date: 2025-04-28T17:10:25Z
#url: https://api.github.com/gists/49e2f53d6bc20640dc6df65df5db8de5
#owner: https://api.github.com/users/KubaO

#! /usr/local/bin/python3.13
#
# Licensed under the MIT license. (c) 2025 Kuba Sunderland-Ober
#
# A GUI tape recorder utility for ABC802 connected to Raspberry Pi GPIO
#
# On the Pi, ensure that pigpiod is running, allowing remote connections.
#
# Edit /etc/systemd/system/pigpiod.service/public.conf to change pigpiod
# command line arguments to:
# /usr/bin/pigpiod -t 0 -b 1000 -s 10 -x 0x18
# -t 0  - use PWM timing source, so that playback speed is correct
# -b 1000 - buffer 1000 ms of data (default: 150)
# -s 10 - sample period 10us (default: 5us)
# -x 0x18 - allow access to gpio 3 and 4 only (0x18 = 1<<3 | 1<<4)
#
# On RPi, to verify what arguments pigpiod was started with, do
# ps -eo args | grep pigpiod
import glob
import ipaddress
import os
import pickle
import re

import pigpio
import platform
import subprocess
import time
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as scrolledtext
from dataclasses import dataclass
from enum import Enum

VERSION = "0.5"

TAPE_IN_GPIO = 4
TAPE_OUT_GPIO = 3

TAPE_LONGEST_GLITCH = 50
TAPE_TIMER_PERIOD = 250

root: tk.Tk | None = None

log_destinations = []


def add_log_destination(dest):
    global log_destinations
    log_destinations.append(dest)


def log(*args, **kwargs):
    for dest in log_destinations:
        dest(*args, **kwargs)


def get_remote_ip():
    """Gets the IP address of the remote Raspberry Pi connected via point-to-point Ethernet on Windows"""
    result = subprocess.run(["arp", "-a"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    _ip = None
    output = result.stdout.decode(encoding='utf-8').splitlines()
    for line in output:
        fields = line.split()
        if fields:
            if fields[0].startswith('169.254') and fields[-1] == 'dynamic':
                _ip = fields[0]

    return _ip and ipaddress.ip_address(_ip)


def stylize(styles: dict):
    style = ttk.Style()
    what = 'foreground' if platform.system() == 'Windows' else 'background'
    for key, value in styles.items():
        if 'button_background' in value:
            value[what] = value['button_background']
            del value['button_background']
        style.configure(key, **value)
    return style


def stylemap(style: ttk.Style, name: str, **kwargs):
    what = 'foreground' if platform.system() == 'Windows' else 'background'
    config = {}
    for key, values in kwargs.items():
        if key == 'button_background':
            key = what
        config[key] = values
    style.map(name, **config)


class DummyIO:
    LOW = 0
    HIGH = 1
    TIMEOUT = 2

    def __init__(self):
        self._callback = None
        self._period = None
        self._gpio = None
        self._busy = False
        self._wave_duration = 0

    def set_mode(self, *args):
        pass

    def write(self, *args):
        pass

    def set_glitch_filter(self, *args):
        pass

    def callback(self, _gpio, _edge, callback):
        class CallbackHandle:
            def __init__(self, _io):
                self.io = _io

            def cancel(self):
                self.io._callback = None

        self._callback = callback
        return CallbackHandle(self)

    def set_watchdog(self, gpio, period):
        self._gpio = gpio
        self._period = period
        root.after(self._period, self._watchdog)

    def _watchdog(self):
        if self._callback is not None:
            tick = self.get_current_tick()
            self._callback(self._gpio, DummyIO.TIMEOUT, tick)
        root.after(self._period, self._watchdog)

    def stop(self):
        pass

    def get_current_tick(self):
        return round(time.monotonic() * 1_000_000)

    def wave_clear(self):
        self._wave_duration = 0

    def wave_add_generic(self, pulses: list[pigpio.pulse]):
        duration = 0
        for p in pulses:
            duration += p.delay
        self._wave_duration += duration

    def wave_create(self):
        return 1

    def wave_send_once(self, _wave_id):
        self._busy = True
        root.after(self._wave_duration // 1000, self._done_sending)

    def _done_sending(self):
        self._busy = False

    def wave_tx_busy(self):
        return self._busy

    def wave_tx_stop(self):
        self._done_sending()

    def wave_delete(self, _id):
        pass


class State(Enum):
    NONE = 0
    IDLE = 1
    RECORDING = 2
    PLAYING = 3


class Tape:
    def __init__(self, io: pigpio.pi):
        self.state = State.NONE
        self.data = []
        self.listeners = []
        self.tk = None
        self.chunks = None
        self.wave_ids = None
        self.record_idle = 0
        self.tape_callback = None
        self.io = io

    def set_tk(self, _tk):
        self.tk = _tk

    def init_io(self):
        # gpio 3 is output to ABC802
        self.io.set_mode(TAPE_OUT_GPIO, pigpio.OUTPUT)
        self.io.write(TAPE_OUT_GPIO, pigpio.LOW)
        # gpio 4 is input from ABC802
        self.io.set_mode(TAPE_IN_GPIO, pigpio.INPUT)
        self.io.set_glitch_filter(TAPE_IN_GPIO, TAPE_LONGEST_GLITCH)

        self.tape_callback = self.io.callback(TAPE_IN_GPIO, pigpio.EITHER_EDGE, self.callback)
        self.io.set_watchdog(TAPE_IN_GPIO, TAPE_TIMER_PERIOD)
        self.callback(TAPE_IN_GPIO, pigpio.TIMEOUT, -1)

    def deinit_io(self):
        self.tape_callback.cancel()

    def play(self, data: list[tuple[int, int]]):
        def make_pulse(_level, _delay):
            if _level == pigpio.HIGH:
                return pigpio.pulse(1 << TAPE_OUT_GPIO, 0, _delay)
            elif _level == pigpio.LOW:
                return pigpio.pulse(0, 1 << TAPE_OUT_GPIO, _delay)

        if self.state != State.IDLE:
            return

        # Gotchas:
        # 1. When too many pulses are sent in one wave_add_generic, the request size overflows and the network
        #    connection closes. The error is 'ConnectionResetError: [Errno 104] Connection reset by peer'
        # 2. When a single wave is too long, 'waveform has too many pulses' error happens
        # 3. When there are too many waves, 'No more CBs for waveform' error happens

        CHUNK_SIZE = 5200
        self.wave_ids = []
        self.chunks = []
        for i0 in range(0, len(data) - 1, CHUNK_SIZE):
            chunk = []
            for i in range(i0, min(i0 + CHUNK_SIZE, len(data) - 1)):
                delay = (data[i + 1][0] - data[i][0])
                level = data[i][1]
                chunk.append(make_pulse(level, delay))
            self.chunks.append(chunk)

        # last pulse gets an aritifical delay of 10ms
        level = data[-1][1]
        self.chunks[-1].append(make_pulse(level, 10000))
        chunk_sizes = " ".join(map(str, map(len, self.chunks)))
        log("* Playback starts")
        log(f"  Chunk sizes: {chunk_sizes}")
        self.io.wave_clear()
        self.wave_ids = []
        self.emit(State.PLAYING, self.io.get_current_tick())

    def _progress_play(self) -> bool:
        # we will be sending one wave and building another
        if self.chunks:
            if len(self.wave_ids) < 2:
                self.io.wave_add_generic(self.chunks.pop(0))
                wave_id = self.io.wave_create_and_pad(50)
                # print(f"+{wave_id}")
                self.wave_ids.append(wave_id)
                cbs = self.io.wave_send_using_mode(wave_id, pigpio.WAVE_MODE_ONE_SHOT_SYNC)
                # print(cbs)
            else:
                if self.io.wave_tx_at() != self.wave_ids[0]:
                    # this waveform has already been sent
                    wave_id = self.wave_ids.pop(0)
                    self.io.wave_delete(wave_id)
                    # print(f"-{wave_id}")
        return len(self.wave_ids) < 2

    def stop(self):
        if self.state == State.PLAYING:
            log("! Playback stopped")
            self.io.wave_tx_stop()

    def callback(self, gpio: int, level: int, tick: int):
        if gpio != TAPE_IN_GPIO:
            print(gpio, level, tick)
            return

        if level == pigpio.TIMEOUT:
            if self.state == State.RECORDING:
                self.record_idle += TAPE_TIMER_PERIOD
                if self.record_idle >= 2000:
                    self.format_data()
                    log("  Recording ends")
                    self.emit(State.IDLE, tick, self.data)
                    self.data = []
            elif self.state == State.NONE:
                self.emit(State.IDLE, tick)
            elif self.state == State.PLAYING:
                while self._progress_play():
                    pass
                if not self.io.wave_tx_busy():
                    log("  Playback ends")
                    self.emit(State.IDLE, tick)
                    if self.wave_ids:
                        for wid in self.wave_ids:
                            self.io.wave_delete(wid)
                    self.wave_ids = None
            return

        self.record_idle = 0

        if self.state != State.RECORDING:
            log("* Recording starts")
            self.emit(State.RECORDING, tick)

        self.data.append((tick, level))

    def emit(self, new_state, *args):
        prev_state = self.state
        event = (new_state, prev_state, *args)
        for listener in self.listeners:
            listener.event(event)
        self.state = new_state

    def add_listener(self, listener):
        self.listeners.append(listener)

    def format_data(self):
        data = self.data
        if not data:
            return

        # every 72 minutes or so, the 32-bit tick overflows
        # correct it by adding 2^32
        prev_tick = 0
        for i in range(len(data)):
            tick, level = data[i]
            if tick < prev_tick:
                tick += 2 ** 32
                data[i] = (tick, level)

        # align timestamps to zero
        zero_tick = data[0][0]
        for i in range(len(data)):
            tick, level = data[i]
            tick -= zero_tick
            data[i] = (tick, level)


def button(master, text, command=None, style="Player.TButton", event=None, **kwargs):
    if event is not None:
        assert command is None
        command = lambda: master.event_generate(event)
    return ttk.Button(master, text=text, style=style, command=command, **kwargs)


class Keyboard(ttk.Frame):
    """On-screen Keyboard"""

    def get(self):
        return self.input.get()

    def set(self, text: str):
        self.input.delete(0, tk.END)
        self.input.insert(0, text)

    def focus(self, kind=None):
        self.input.focus()
        if kind is not None:
            self.kind.configure(text=kind)

    def __init__(self, master):
        super().__init__(master)
        self.root = self.winfo_toplevel()

        font = ("Arial", 20)
        stylize({
            "TLabel": {'font': font},
            "Key.TButton": {'font': font},
            "Enter.TButton": {'button_background': "#14AE5C", 'font': font},
            "BS.TButton": {'button_background': "#EC221F", 'font': font},
            "Cancel.TButton": {'button_background': "#E8B931", 'font': font},
        })
        TEntry_options = {'font': ("Arial", 32)}

        self.kind = ttk.Label(self, anchor=tk.CENTER)
        self.kind.place(x=10, y=10, width=60, height=60)

        self.input = ttk.Entry(self, **TEntry_options)
        self.input.place(x=80, y=10, width=640, height=60)

        def make_key(_char, x, y, width=60):
            def _key():
                self.input.insert(tk.END, _char)
                self.input.focus()

            _button = ttk.Button(self, text=_char, style="Key.TButton", command=_key)
            _button.place(x=x, y=y, width=width, height=60)

        def _enter():
            self.event_generate("<<Accept>>")
            self.input.focus()

        enter = ttk.Button(self, text="⏎", style="Enter.TButton", command=_enter)
        enter.place(x=730, y=10, width=60, height=60)

        for i, char in enumerate("1234567890"):
            make_key(char, x=0 + i * 80, y=90)
        for i, char in enumerate("QWERTYUIOP"):
            make_key(char, x=10 + i * 80, y=170)
        for i, char in enumerate("ASDFGHJKL"):
            make_key(char, x=30 + i * 80, y=250)
        for i, char in enumerate("ZXCVBNM,."):
            make_key(char, x=50 + i * 80, y=330)
        make_key(" ", x=90, y=410, width=480)
        make_key("-", x=590, y=410)

        def _cancel():
            self.event_generate("<<Reject>>")
            self.input.focus()

        cancel = ttk.Button(self, text="×", style="Cancel.TButton", command=_cancel)
        cancel.place(x=10, y=410, width=60, height=60)

        def _bs():
            length = len(self.input.get())
            if length:
                self.input.delete(length - 1, tk.END)
            self.input.focus()

        bs = ttk.Button(self, text="⌫", style="BS.TButton", command=_bs)
        bs.place(x=670, y=410, width=124, height=60)


class Popup(object):
    def __init__(self, parent: tk.Tk, **kwargs):
        self.result = False

        self.root = tk.Toplevel(parent)
        self.root.geometry("480x280+160+100")
        self.root.transient(parent)
        self.root.title(kwargs.get('title', ""))
        self.root.resizable(width=False, height=False)

        font = ("Arial", 20)
        stylize({
            "Icon.Popup.TLabel": {'font': font},
            "Popup.TLabel": {'font': ('Arial', 16)},
            "Yes.Popup.TButton": {'button_background': '#14AE5C', 'font': font},
            "No.Popup.TButton": {'button_background': '#EC221F', 'font': font},
        })
        message = kwargs.get('message', "")

        frame = tk.Frame(self.root)
        frame.place(relwidth=1, relheight=1)

        question = ttk.Label(frame, text="❔", style="Icon.Popup.TLabel")
        question.place(x=20, y=20, width=40, height=60)
        message = ttk.Label(frame, text=message, justify=tk.CENTER, style="Popup.TLabel", wraplength=360)
        message.place(x=80, y=20, width=380, height=160)
        no = ttk.Button(frame, text="❌", style="No.Popup.TButton", command=self._no)
        no.place(x=170, y=200, width=80, height=60)
        yes = ttk.Button(frame, text="✔", style="Yes.Popup.TButton", command=self._yes)
        yes.place(x=290, y=200, width=80, height=60)

        self.root.wait_visibility()
        self.root.grab_set()

    def _done(self, result):
        self.result = result
        self.root.grab_release()
        self.root.destroy()

    def _yes(self):
        self._done(True)

    def _no(self):
        self._done(False)

    @staticmethod
    def askyesno(**kwargs):
        popup = Popup(root, **kwargs)
        root.wait_window(popup.root)
        return popup.result


class Player(ttk.Frame):
    """Player Pane"""

    def focus(self, *args) -> str | None:
        if args:
            self.tree.selection_set(*args)
            self.tree.focus(args[0])
            self.tree.see(args[0])
        else:
            return self.tree.focus()

    def add_row(self, text, *columns):
        return self.tree.insert('', 0,
                                text=text, values=columns)

    def remove_row(self, item: str):
        self.tree.delete(item)

    def update_row(self, item: str, text, length, time):
        self.tree.item(item, text=text)
        self.tree.set(item, 'length', length)
        self.tree.set(item, 'time', time)

    def log_add(self, line: str, end="\n"):
        self.log.insert(tk.END, line + end)
        self.log.see(tk.END)

    def __init__(self, master):
        super().__init__(master)
        self.root = self.winfo_toplevel()
        font = ("Arial", 20)
        style = stylize({
            "Player.TLabel": {'font': ("Arial", 13)},
            "Player.TButton": {'font': font},
            "Treeview": {'font': ('Arial', 14), 'rowheight': 60},
            "Treeview.Heading": {'font': ('Arial', 14)},
            "Mode.Player.TButton": {'font': font},
            "Play.Player.TButton": {'button_background': "#A9E5C5", 'font': font},
            "Vertical.TScrollbar": {'arrowsize': 80, 'width': 40},
        })
        stylemap(style, 'Play.Player.TButton', button_background=[
            ('alternate', "#DD9F9F")
        ])

        self.log = scrolledtext.ScrolledText(self)
        self.log.vbar.configure(width=40)
        self.log.place(x=20, y=20, width=540, height=440)

        self.tree_scroll = ttk.Scrollbar(self)
        self.tree_scroll.place(x=520, y=20, width=40, height=440)

        self.tree = ttk.Treeview(self, columns=("length", "time"), yscrollcommand=self.tree_scroll.set)
        self.tree_scroll.config(command=self.tree.yview)
        self.tree.heading("length", text="⏳")
        self.tree.heading("time", text="Time")
        self.tree.column("#0", width=0, stretch=tk.YES)
        self.tree.column("#1", width=80, stretch=tk.NO)
        self.tree.column("#2", width=170, stretch=tk.NO)
        self.tree.place(x=20, y=20, width=500, height=440)

        self.tree.bind("<<TreeviewSelect>>", lambda ev: self.event_generate("<<ItemSelected>>"))

        version = ttk.Label(self, text=f"Ver\n{VERSION}", style="Player.TLabel", anchor=tk.CENTER)
        version.place(x=700, y=320, width=60, height=60)

        def _mode():
            if self.mode['text'] == "Log":
                self.log.frame.tkraise()
                self.mode['text'] = "List"
            else:
                self.tree.tkraise()
                self.tree_scroll.tkraise()
                self.mode['text'] = "Log"

        self.mode = button(self, text="Log", style="Mode.Player.TButton", command=_mode)
        self.mode.place(x=600, y=20, width=80, height=60)

        _exit = button(self, text="❌", command=self.root.destroy)
        _exit.place(x=720, y=20, width=60, height=60)

        self.play = button(self, text="⏯", style="Play.Player.TButton", event="<<Play_Stop>>")
        self.play.place(x=600, y=100, width=180, height=100)

        self.previous = button(self, text="⏮", event="<<Previous>>")
        self.previous.place(x=600, y=220, width=80, height=60)

        self.next = button(self, text="⏭", event="<<Next>>")
        self.next.place(x=700, y=220, width=80, height=60)

        self.remove = button(self, text="🗑", event="<<Remove>>")
        self.remove.place(x=700, y=400, width=80, height=60)

        self.search = button(self, text="🔍", event="<<Search>>")
        self.search.place(x=600, y=320, width=80, height=60)

        self.info = button(self, text="ℹ", event="<<Info>>")
        self.info.place(x=600, y=400, width=80, height=60)

    def event(self, event):
        match event:
            case (State.RECORDING, *_):
                self.play.configure(text="⏺")
                self.play.state([tk.DISABLED, 'alternate'])
                self.previous['state'] = tk.DISABLED
                self.next['state'] = tk.DISABLED

            case (State.PLAYING, *_):
                self.play.configure(text="⏹", command=lambda: self.event_generate("<<Stop>>"))
                self.play.state(['!alternate'])
                self.play['state'] = tk.NORMAL
                self.previous['state'] = tk.DISABLED
                self.next['state'] = tk.DISABLED

            case (State.IDLE, *_):
                self.play.configure(text="⏯", command=lambda: self.event_generate("<<Play>>"))
                self.play.state(['!alternate'])
                self.play['state'] = tk.NORMAL
                self.previous['state'] = tk.NORMAL
                self.next['state'] = tk.NORMAL


@dataclass
class Recording:
    title: str
    length: float  # [seconds]
    time: str
    data: list[(int, int)]  # [(tick, level), ...]

    def as_row(self):
        return (self.title,
                f"{self.length:0.2f}", self.time.replace(" ", "\n"))


class Gui:
    def show_player(self):
        self.notebook.select(1)

    def show_keyboard(self, kind=None):
        self.notebook.select(0)
        self.keyboard.focus(kind=kind)

    def __init__(self, master, ip):
        self.master = master
        self.tape = None
        self.ip = ip

        self.recordings = []
        self.recordings_index = {}
        self.edited_item = None
        self.searching = False

        master.resizable(width=False, height=False)
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        style = ttk.Style()
        style.layout("Tabless.TNotebook.Tab", [])
        notebook = ttk.Notebook(master, style="Tabless.TNotebook")
        notebook.grid(column=0, row=0, sticky=tk.NSEW)
        notebook.columnconfigure(0, weight=1)
        notebook.rowconfigure(0, weight=1)

        self.keyboard = Keyboard(notebook)
        notebook.add(self.keyboard)

        self.player = Player(notebook)
        notebook.add(self.player)

        add_log_destination(self.player.log_add)
        self.notebook = notebook
        self.show_player()

        self.player.bind("<<Play>>", self.play)
        self.player.bind("<<Stop>>", self.stop)
        self.player.bind("<<Next>>", self.next)
        self.player.bind("<<Previous>>", self.previous)
        self.player.bind("<<Info>>", self.info)
        self.player.bind("<<Search>>", self.search)
        self.player.bind("<<Remove>>", self.remove)

        self.keyboard.bind("<<Accept>>", self.keyboard_accept)
        self.keyboard.bind("<<Reject>>", self.keyboard_done)

        log(f"- Expecting Connections:")
        log(f"  ABC802 Tape -> RPi GPIO{TAPE_IN_GPIO}")
        log(f"  ABC802 Tape <- RPi GPIO{TAPE_OUT_GPIO}")

        self._load()

    def set_tape(self, tape: Tape):
        self.tape = tape
        tape.set_tk(self)
        tape.add_listener(self)
        tape.add_listener(self.player)

        if isinstance(self.tape.io, DummyIO):
            log("! Cannot connect to RPi, using dummy I/O")
        else:
            log(f"* Connected to {self.ip}")

    def _load(self):
        files = glob.glob(os.path.expanduser("~/*.tape"))
        for filename in files:
            log(f"* Loading {filename}")
            try:
                with open(filename, "rb") as f:
                    recording = pickle.load(f)
                title = re.fullmatch('.*/([^/]*)\\.tape', filename)
                if title:
                    recording.title = title.group(1)
                self._add_recording(recording)
            except:
                log(f"! Loading Failed")
        if not files:
            log(". No Recordings Exist")

    def _save(self, filename, recording) -> bool:
        filename = os.path.expanduser(f"~/{filename}")
        log(f"* Saving {filename}")
        try:
            with open(filename + ".new", "wb") as f:
                pickle.dump(recording, f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                os.remove(filename)
            except:
                pass
            os.rename(filename + ".new", filename)
            os.system("sync")
            return True
        except:
            log(f"! Saving Failed")
            return False

    def _erase(self, filename) -> bool:
        filename = os.path.expanduser(f"~/{filename}")
        log(f"* Erasing {filename}")
        try:
            os.remove(filename)
            return True
        except:
            log(f"! Erasing Failed")
            return False

    def event(self, event):
        # events sent by the tape
        match event:
            case (State.IDLE, State.RECORDING, _tick, data):
                self._new_recording(data)

            case (State.IDLE, *_):
                log(f"* Idle")

    def play(self, _event):
        if not self.recordings:
            log("! No Recordings")
            return
        item = self.player.focus()
        self.tape.play(self.recordings_index[item].data)

    def stop(self, _event):
        self.tape.stop()

    def _new_recording(self, data):
        assert data
        length, _ = data[-1]
        length /= 1_000_000
        when = time.strftime("%Y-%m-%d %I:%M%p")
        recording = Recording("", length, when, data)

        n = len(self.recordings)
        log(f"  Recording #{n}: {length:.3}s, events: {len(data)}")
        recording.title = f"recording {n}"
        if self._save(f"{recording.title}.tape", recording):
            self._add_recording(recording)

    def _add_recording(self, recording: Recording):
        if hasattr(recording, 'length_us'):
            recording.length = recording.length_us
            delattr(recording, 'length_us')
        item = self.player.add_row(*recording.as_row())

        self.recordings.insert(0, recording)
        self.recordings_index[item] = recording
        self.player.focus(item)

    def _remove_recording(self, item: str):
        self.player.remove_row(item)
        recording = self.recordings_index.pop(item)
        self.recordings.remove(recording)

    def next(self, _event):
        cur_item = self.player.focus()
        if cur_item:
            next_item = None
            for item in self.recordings_index.keys():
                if item == cur_item:
                    break
                next_item = item
            if next_item:
                self.player.focus(next_item)

    def previous(self, _event):
        cur_item = self.player.focus()
        if cur_item:
            ready = False
            for item in self.recordings_index.keys():
                if ready:
                    self.player.focus(item)
                    break
                ready = item == cur_item

    def search(self, _event):
        self.show_keyboard(kind='🔍')
        self.keyboard.set("")
        self.searching = True

    def info(self, _event):
        item = self.player.focus()
        if not item:
            return
        self.show_keyboard(kind='ℹ')
        recording = self.recordings_index[item]
        self.edited_item = item
        self.keyboard.set(recording.title)

    def keyboard_accept(self, event):
        if self.searching:
            self.search_accept(event)
        else:
            self.edit_accept(event)

    def edit_accept(self, _event):
        item = self.edited_item
        assert item
        recording = self.recordings_index[item]
        new_title = self.keyboard.get()
        old_title = recording.title
        if old_title != new_title:
            recording.title = new_title
            if self._save(f"{new_title}.tape", recording):
                self._erase(f"{old_title}.tape")
            else:
                recording.title = old_title
            self.player.update_row(item, *recording.as_row())
        self.keyboard_done(None)

    def search_accept(self, _event):
        needle = self.keyboard.get().upper()
        items = []
        for item, recording in self.recordings_index.items():
            if needle in recording.title:
                items.append(item)
        self.player.focus(*items)
        self.keyboard_done(None)

    def keyboard_done(self, _event):
        self.edited_item = None
        self.searching = False
        self.show_player()

    def remove(self, _event):
        item = self.player.focus()
        if not item:
            return
        recording = self.recordings_index[item]
        message = f"Are you sure you want to delete '{recording.title}', {recording.length:.2f}s long?"
        if Popup.askyesno(title="Delete Recording", message=message):
            if self._erase(f"{recording.title}.tape"):
                self._remove_recording(item)


def main():
    global root

    if platform.system() == 'Windows':
        # make it look nice
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)

    root = tk.Tk()

    if platform.system() == 'Windows':
        # remote connection
        ip = get_remote_ip()
        if ip is None:
            io = DummyIO()
        else:
            io = pigpio.pi(str(ip))
    else:
        # local connection
        ip = "local gpiod"
        io = pigpio.pi()

    if ip is not None:
        root.title(f"Tape ({ip})")
    else:
        root.title("Tape")

    root.geometry("800x480")
    if platform.system() != 'Windows':
        root.attributes("-fullscreen", True)

    _gui = Gui(root, ip)
    tape = Tape(io)
    _gui.set_tape(tape)
    tape.init_io()

    root.mainloop()

    tape.deinit_io()
    io.stop()


if __name__ == '__main__':
    main()
