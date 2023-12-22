#date: 2023-12-22T16:52:53Z
#url: https://api.github.com/gists/7fcd0ce04003c804491e8e456af1dd4a
#owner: https://api.github.com/users/av1d

import sys
import time
import webview


def getInput(window):
    while True:
        getKeys = input('Press x to exit? ')
        if getKeys.lower() == 'x':
            window.destroy()
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    window = webview.create_window('Destroy', 'https://example.org/')
    webview.start(getInput, window)
