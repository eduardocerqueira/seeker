#date: 2022-09-20T17:17:56Z
#url: https://api.github.com/gists/cd7eb0a9ac114f4ec62d1e3a0efe931c
#owner: https://api.github.com/users/lewars

# -*- coding: utf-8 -*-

import re
from xkeysnail.transform import *

# define timeout for multipurpose_modmap
define_timeout(1)

# # [Global modemap] Change modifier keys as in xmodmap
# define_modmap({
#     Key.CAPSLOCK: Key.LEFT_CTRL
# })

# # [Conditional modmap] Change modifier keys in certain applications
# define_conditional_modmap(re.compile(r'Emacs'), {
#     Key.RIGHT_CTRL: Key.ESC,
# })

# # [Multipurpose modmap] Give a key two meanings. A normal key when pressed and
# # released, and a modifier key when held down with another key. See Xcape,
# # Carabiner and caps2esc for ideas and concept.
# define_multipurpose_modmap(
#     # Enter is enter when pressed and released. Control when held down.
#     {Key.ENTER: [Key.ENTER, Key.RIGHT_CTRL]}

#     # Capslock is escape when pressed and released. Control when held down.
#     # {Key.CAPSLOCK: [Key.ESC, Key.LEFT_CTRL]
#     # To use this example, you can't remap capslock with define_modmap.
# )

# # [Conditional multipurpose modmap] Multipurpose modmap in certain conditions,
# # such as for a particular device.
# define_conditional_multipurpose_modmap(lambda wm_class, device_name: device_name.startswith("Microsoft"), {
#    # Left shift is open paren when pressed and released.
#    # Left shift when held down.
#    Key.LEFT_SHIFT: [Key.KPLEFTPAREN, Key.LEFT_SHIFT],

#    # Right shift is close paren when pressed and released.
#    # Right shift when held down.
#    Key.RIGHT_SHIFT: [Key.KPRIGHTPAREN, Key.RIGHT_SHIFT]
# })

# Keybindings for Abricotine
define_keymap(re.compile("abricotine", flags=re.IGNORECASE), {

    # Newline
    K("C-m"): K("enter"),
    K("C-o"): [K("enter"), K("left")],
    # Copy
    K("C-y"): [K("C-v"), set_mark(False)],
    # Non-Emacs cut, copy, and paste
    K("Super-x"): K("C-x"),
    K("Super-c"): K("C-c"),
    K("Super-v"): K("C-v"),
    K("Super-Shift-v"): K("C-Shift-v"),
    # Undo
    K("Super-z"): [K("C-z"), set_mark(False)],
    K("Super-Shift-z"): [K("C-y"), set_mark(False)],
    # Mark
    K("C-space"): set_mark(True),
    K("C-M-space"): with_or_set_mark(K("C-right")),
    # Search
    K("C-s"): K("F3"),
    K("C-r"): K("Shift-F3"),
    # Non-Emacs Search
    K("Super-f"): K("C-f"),
    K("Super-g"): K("Shift-F3"),
    K("Super-Shift-g"): K("C-Shift-g"),
    # Non-Emacs Print
    K("Super-p"): K("C-p"),
    # Cancel
    K("C-g"): [K("esc"), set_mark(False)],
    # Italic, Bold, Strike-through
    K("Super-i"): K("C-i"),
    K("Super-b"): K("C-b"),

    # C-x YYY
    K("C-x"): {
    #     # C-x h (select all)
    #     K("h"): [K("C-home"), K("C-a"), set_mark(True)],
    #     # C-x C-f (open)
    #     K("C-f"): K("C-o"),
    #     # C-x C-s (save)
    #     K("C-s"): K("C-s"),
    #     # C-x k (kill tab)
    #     K("k"): K("C-f4"),
    #     # C-x C-c (exit)
    #     K("C-c"): K("C-q"),
    #     # cancel
    #     K("C-g"): pass_through_key,
    #     # C-x u (undo)
    #     K("u"): [K("C-z"), set_mark(False)],
    }

}, "Abricotine")

# Keybindings for Firefox/Chrome
define_keymap(re.compile("Firefox|Google-chrome|Brave-browser", flags=re.IGNORECASE), {
    # Tab
    K("Super-t"): K("C-t"), # open tab
    K("Super-w"): K("C-w"), # close tab
    K("Super-Shift-RIGHT_BRACE"): K("C-TAB"),
    K("Super-Shift-LEFT_BRACE"): K("C-Shift-TAB"),

    # # Type C-j to focus to the content
    # K("C-j"): K("C-f6"),
    # # very naive "Edit in editor" feature (just an example)
    # K("C-o"): [K("C-a"), K("C-c"), launch(["gedit"]), sleep(0.5), K("C-v")]

    # Page up/down
    K("M-v"): with_mark(K("page_up")),
    # K("C-v"): with_mark(K("page_down")),
    # Newline
    K("C-m"): K("enter"),
    K("C-j"): K("enter"),
    K("C-o"): [K("enter"), K("left")],
    # Copy
    # K("C-w"): [K("C-x"), set_mark(False)],
    # K("M-w"): [K("C-c"), set_mark(False)],
    K("C-y"): [K("C-v"), set_mark(False)],
    # Non-Emacs cut, copy, and paste
    K("Super-x"): K("C-x"),
    K("Super-c"): K("C-c"),
    K("Super-v"): K("C-v"),
    K("Super-Shift-v"): K("C-Shift-v"),
    # Delete
    K("C-d"): [K("delete"), set_mark(False)],
    K("M-d"): [K("C-delete"), set_mark(False)],
    # K("C-slash"): [K("C-z"), set_mark(False)],
    # K("C-Shift-ro"): K("C-z"),
    # Mark
    K("C-space"): set_mark(True),
    K("C-M-space"): with_or_set_mark(K("C-right")),
    # Search
    K("C-s"): K("F3"),
    K("C-r"): K("Shift-F3"),
    K("M-Shift-key_5"): K("C-h"),
    # Non-Emacs Search
    K("Super-f"): K("C-f"),
    K("Super-g"): K("C-g"),
    K("Super-Shift-g"): K("C-Shift-g"),
    # Non-Emacs Print
    K("Super-p"): K("C-p"),
    # Non-Emacs Reload
    K("Super-r"): K("C-r"),
    K("Super-Shift-r"): K("C-Shift-r"),
    # Cancel
    K("C-g"): [K("esc"), set_mark(False)],
    # Escape
    K("C-q"): escape_next_key,

    # New Window/Incognito Window
    K("Super-n"): K("C-n"),
    K("Super-Shift-n"): K("C-Shift-n"),
}, "Firefox and Chrome")

# Global keybindings
define_keymap(lambda wm_class: wm_class not in ("Emacs", "gnome-terminal-server", "Gnome-terminal", "gnome-boxes"), {
    # Cut, copy, and paste
    K("Super-x"): K("C-x"),
    K("Super-c"): K("C-c"),
    K("Super-v"): K("C-v"),
    # Cursor
    K("C-b"): with_mark(K("left")),
    K("C-f"): with_mark(K("right")),
    K("C-p"): with_mark(K("up")),
    K("C-n"): with_mark(K("down")),
    # Forward/Backward word
    K("M-b"): with_mark(K("C-left")),
    K("M-f"): with_mark(K("C-right")),
    # Beginning/End of line
    K("C-a"): with_mark(K("home")),
    K("C-e"): with_mark(K("end")),
    # Newline
    K("C-o"): [K("enter"), K("left")],
    # Undo
    K("Super-z"): [K("C-z"), set_mark(False)],
    K("Super-Shift-z"): [K("C-y"), set_mark(False)],
    # Copy
    K("C-y"): [K("C-v"), set_mark(False)],
    # Kill line
    K("C-k"): [K("Shift-end"), K("C-x"), set_mark(False)],
    # Quit
    K("Super-q"): K("C-q"),
}, "Global Keybindings")