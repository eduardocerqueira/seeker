#date: 2022-10-07T17:26:33Z
#url: https://api.github.com/gists/e9b6e5ea285f60cd822801bd25619b6c
#owner: https://api.github.com/users/r0oland

# ------------------------------------------------------------------------------
def color_message(str, color = 'red', effects=['bold'], bgColor = ''):
    '''Short warning...it makes a short warning. Really.'''
    
    # create dictionary with all colors and corresponding escape codes
    COLORS = {
        'black': "\x1b[30m",
        'red': "\x1b[31m",
        'green': "\x1b[32m",
        'yellow': "\x1b[33m",
        'blue': "\x1b[34m",
        'magenta': "\x1b[35m",
        'cyan': "\x1b[36m",
        'white': "\x1b[37m",
        'meterblue': "\x1b[38;5;4m",
        'grey': "\x1b[38;5;8m",
        'gray': "\x1b[38;5;8m",
        'dark_red': "\x1b[38;5;1m",
        'purple': "\x1b[38;5;5m",
        'orange': "\x1b[38;5;202m",
        '': ""
    }
    # see https://en.wikipedia.org/wiki/ANSI_escape_code for even more colors

    # create dict with background colors
    BG_COLORS = {
        "black": "\x1b[40m",
        "red": "\x1b[41m",
        "green": "\x1b[42m",
        "yellow": "\x1b[43m",
        "blue": "\x1b[44m",
        "magenta": "\x1b[45m",
        "cyan": "\x1b[46m",
        "white": "\x1b[47m",
        "": ""
    }

    EFFECTS = {
        "bold": "\x1b[1m",
        "underscore": "\x1b[4m",
        "reverse": "\x1b[7m",
        "hidden": "\x1b[8m",
        "": ""
    }

    RESET = "\x1b[0m"

    allEffects = ''
    for effect in effects:
        # print(effect)
        allEffects = allEffects + EFFECTS[effect]

    formatString = allEffects + BG_COLORS[bgColor] + COLORS[color]

    print(formatString + "{:s}".format(str) + RESET, end='')