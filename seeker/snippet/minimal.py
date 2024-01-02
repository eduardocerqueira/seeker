#date: 2024-01-02T17:01:21Z
#url: https://api.github.com/gists/726e1780bf6cec8edef6352bc32659b0
#owner: https://api.github.com/users/RhetTbull

"""Implements a minimalist Cocoa application using pyobjc

Based on the example found here:
https://www.cocoawithlove.com/2010/09/minimalist-cocoa-programming.html

To run this: 
    - save as minimal.py
    - pip3 install pyobjc
    - python3 minimal.py
"""

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSMakeRect,
    NSMenu,
    NSMenuItem,
    NSProcessInfo,
    NSTitledWindowMask,
    NSWindow,
)
from Foundation import NSMakePoint


def main():
    """Create a minimalist window programmatically, without a NIB file."""
    with objc.autorelease_pool():
        # create the app
        NSApplication.sharedApplication()
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        # create the menu bar and attach it to the app
        menubar = NSMenu.alloc().init().autorelease()
        app_menu_item = NSMenuItem.alloc().init().autorelease()
        menubar.addItem_(app_menu_item)
        NSApp.setMainMenu_(menubar)
        app_menu = NSMenu.alloc().init().autorelease()

        # add a menu item to the menu to quit the app
        app_name = NSProcessInfo.processInfo().processName()
        quit_title = f"Quit {app_name}"
        quit_menu_item = (
            NSMenuItem.alloc()
            .initWithTitle_action_keyEquivalent_(quit_title, "terminate:", "q")
            .autorelease()
        )
        app_menu.addItem_(quit_menu_item)
        app_menu_item.setSubmenu_(app_menu)

        # create the window
        window = (
            NSWindow.alloc()
            .initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 200, 200),
                NSTitledWindowMask,
                NSBackingStoreBuffered,
                False,
            )
            .autorelease()
        )
        window.cascadeTopLeftFromPoint_(NSMakePoint(20, 20))
        window.setTitle_(app_name)
        window.makeKeyAndOrderFront_(None)

        # run the app
        NSApp.activateIgnoringOtherApps_(True)
        NSApp.run()
        return 0


if __name__ == "__main__":
    main()
