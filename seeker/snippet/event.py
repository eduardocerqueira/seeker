#date: 2022-09-16T22:10:19Z
#url: https://api.github.com/gists/1b8b491ca7f2d4c3fa82a2cbf4d0c282
#owner: https://api.github.com/users/miezebieze

import pygame
from collections import namedtuple
from typing import Optional, Iterator, NamedTuple, AnyStr, Sequence

# edit: examples are at the bottom

_raw  = {
        pygame.QUIT             : "",
        pygame.ACTIVEEVENT      : "gain state",
        pygame.VIDEORESIZE      : "size w h",
        pygame.VIDEOEXPOSE      : "",

        pygame.KEYDOWN          : "key mod unicode scancode window",
        pygame.KEYUP            : "key mod unicode scancode window",

        pygame.MOUSEMOTION      : "pos rel buttons touch window",
        pygame.MOUSEBUTTONUP    : "pos button touch window",
        pygame.MOUSEBUTTONDOWN  : "pos button touch window",
        pygame.MOUSEWHEEL       : "which flipped x y touch window",

        pygame.FINGERMOTION     : "touch_id finger_id x y dx dy pressure",
        pygame.FINGERDOWN       : "touch_id finger_id x y dx dy pressure",
        pygame.FINGERUP         : "touch_id finger_id x y dx dy pressure",
        pygame.MULTIGESTURE     : "touch_id x y pinched rotated num_fingers",

        pygame.JOYAXISMOTION    : "joy instance_id axis value",
        pygame.JOYBALLMOTION    : "joy instance_id ball rel",
        pygame.JOYHATMOTION     : "joy instance_id hat value",
        pygame.JOYBUTTONUP      : "joy instance_id button",
        pygame.JOYBUTTONDOWN    : "joy instance_id button",

        pygame.TEXTEDITING      : "text start length",
        pygame.TEXTINPUT        : "text window",

        pygame.DROPBEGIN        : "",
        pygame.DROPCOMPLETE     : "",
        pygame.DROPFILE         : "file",
        pygame.DROPTEXT         : "text",

        pygame.MIDIIN           : "",
        pygame.MIDIOUT          : "",
        pygame.AUDIODEVICEADDED         : "which iscapture",
        pygame.AUDIODEVICEREMOVED       : "which iscapture",

        pygame.CONTROLLERDEVICEADDED    : "device_index",
        pygame.JOYDEVICEADDED           : "device_index",
        pygame.CONTROLLERDEVICEREMOVED  : "instance_id",
        pygame.JOYDEVICEREMOVED         : "instance_id",
        pygame.CONTROLLERDEVICEREMAPPED : "instance_id",

        pygame.USEREVENT        : "code",

        pygame.WINDOWSHOWN      : "window",
        pygame.WINDOWHIDDEN     : "window",
        pygame.WINDOWEXPOSED    : "window",
        pygame.WINDOWMOVED      : "window x y",
        pygame.WINDOWRESIZED    : "window x y",
        pygame.WINDOWSIZECHANGED: "window x y",
        pygame.WINDOWMINIMIZED  : "window",
        pygame.WINDOWMAXIMIZED  : "window",
        pygame.WINDOWRESTORED   : "window",
        pygame.WINDOWENTER      : "window",
        pygame.WINDOWLEAVE      : "window",
        pygame.WINDOWFOCUSGAINED: "window",
        pygame.WINDOWFOCUSLOST  : "window",
        pygame.WINDOWCLOSE      : "window",
        pygame.WINDOWTAKEFOCUS  : "window",
        pygame.WINDOWHITTEST    : "window",
        }

class Event :
    _ignore: Sequence [AnyStr] = ()

    def __init__ (self, ignore: Sequence [AnyStr] = ()) :
        _it = list ()
        for i in self._ignore + ignore :
            try:
                _it.append (getattr (pygame, i.upper ()))
            except AttributeError :
                print (f"'{i.upper ()}' not a pygame event type")
        self._ignoretypes = tuple (_it)

        # building the namedtuples that pygame Events are replaced with
        self._table = dict ()
        for type, keys in _raw.items ():
            name = pygame.event.event_name (type)
            evtup = namedtuple (name, keys.split ())
            setattr (self, name, evtup)
            self._table[type] = evtup

    def get (self) -> Iterator [NamedTuple] :
        for event in pygame.event.get (exclude = self._ignoretypes) :
            try :
                yield self._table[event.type] ( **event.dict)
            except KeyError :
                print (f"'{pygame.event.event_name (event.type)}':",
                        "does not the implemented!")


# you can safely delete everything below the previous line.
# examples start here (i put them into functions so the file can be safely imported.)
def classical_example () :
    class Controls (Event) :
        _ignore = ("windowmoved", "WindowMoved")
        quit_signal = False
        drag = None  # tuple of movement per tick
        mouse = None
        mouse_moved = False

        def handle_events (self) :
            self.drag = None
            self.mouse_moved = False
            for e in self.get () :
                match e :
                    case self.Quit () :
                        self.quit_signal = True
                    case self.MouseMotion (buttons=(0, 0, 0), pos=(x, y), rel=(dx, dy)) :
                        self.mouse = x, y
                        self.mouse_moved = True
                        print ("pure motion")
                    case self.MouseMotion (buttons=(1, _, _), pos=(x, y), rel=(dx, dy)) :
                        self.drag = dx, dy
                        print ("button 1 drag")

    class Game :
        keep_running = True
        def __init__ (self) :
            self.controls = Controls ()

        def loop (self) :
            while self.keep_running :
                self.controls.handle_events ()

                if self.controls.quit_signal :
                    self.keep_running = False
                if self.controls.drag is not None :
                    ... # drag something around
                if self.controls.mouse_moved :
                    ... # better check if something is hovered

                # tickle objects
                # draw your junk or whatnot


def functional_example () :
    controls = Event (ignore=("windowmoved", "WindowMoved"))
    keep_running = True
    position = [0, 0]

    while keep_running :
        for e in controls.get () :
            match e :
                case controls.Quit () :
                    keep_running = False
                case controls.KeyDown (key=pygame.K_c, mod=pygame.KMOD_LCTRL) :
                    keep_running = False

                case controls.KeyDown (unicode="a") : position[0] -= 1
                case controls.KeyDown (unicode="d") : position[0] += 1
                case _ : print (e)
        # do other stuff
        # draw something will ya?
