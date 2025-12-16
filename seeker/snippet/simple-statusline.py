#date: 2025-12-16T17:00:32Z
#url: https://api.github.com/gists/3728d8efd4dd62ad82b22e0e286e85d0
#owner: https://api.github.com/users/benabraham

#!/usr/bin/env python3
"""
Simple Claude Code StatusLine Script
Shows context usage/progress with colored bar
Uses current_usage field for accurate context window calculations
"""

import json
import os
import subprocess
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Progress bar ratio (1.0 = 100 chars, 0.25 = 25 chars)
BAR_RATIO = 0.25

# Active theme: 'dark' or 'light'
THEME = 'dark'

# Color format: ((R, G, B), fallback_256)
# Set RGB to None to always use 256 fallback

THEMES = {
    'dark': {
        # Model badge colors: (bg, fg)
        'model_sonnet': (((163, 190, 140), 108), ((46, 52, 64), 236)),   # green bg, dark fg
        'model_opus':   (((136, 192, 208), 110), ((46, 52, 64), 236)),   # aqua bg, dark fg
        'model_haiku':  (((76, 86, 106), 60),    ((216, 222, 233), 253)), # grey bg, light fg
        'model_default':(((216, 222, 233), 253), ((46, 52, 64), 236)),   # light bg, dark fg

        # Unused portion of progress bar
        'bar_empty': (None, 235),  # 256-only, dark grey

        # Text colors
        'text_percent': (((191, 189, 182), None), 250),  # light grey
        'text_numbers': (((191, 189, 182), None), 250),  # light grey
        'text_cwd':     (((216, 222, 233), None), 253),  # snow white
        'text_git':     (((163, 190, 140), None), 108),  # green

        # Progress bar gradient: (threshold, (rgb, fallback_256))
        # Threshold means "use this color if pct < threshold"
        'gradient': [
            (10,  ((24, 53, 34),   22)),   # 0-9%   dark green
            (20,  ((21, 62, 33),   22)),   # 10-19%
            (30,  ((16, 70, 32),   28)),   # 20-29%
            (40,  ((11, 78, 28),   28)),   # 30-39%
            (50,  ((6, 87, 22),    34)),   # 40-49% bright green
            (60,  ((46, 89, 0),    106)),  # 50-59% yellow-green
            (70,  ((93, 79, 0),    136)),  # 60-69% olive
            (80,  ((131, 58, 0),   166)),  # 70-79% orange
            (90,  ((161, 7, 0),    160)),  # 80-89% red-orange
            (101, ((179, 0, 0),    196)),  # 90-100% red
        ],
    },

    'light': {
        # Model badge colors: (bg, fg)
        'model_sonnet': (((143, 170, 120), 107), ((255, 255, 255), 231)),  # muted green bg, white fg
        'model_opus':   (((106, 162, 178), 73),  ((255, 255, 255), 231)),  # muted aqua bg, white fg
        'model_haiku':  (((140, 150, 170), 103), ((255, 255, 255), 231)),  # muted grey bg, white fg
        'model_default':(((100, 110, 130), 66),  ((255, 255, 255), 231)),  # slate bg, white fg

        # Unused portion of progress bar
        'bar_empty': (None, 252),  # 256-only, light grey

        # Text colors
        'text_percent': (((80, 80, 80), None), 240),    # dark grey
        'text_numbers': (((80, 80, 80), None), 240),    # dark grey
        'text_cwd':     (((60, 70, 90), None), 238),    # dark slate
        'text_git':     (((80, 140, 80), None), 65),    # muted green

        # Progress bar gradient
        'gradient': [
            (10,  ((34, 120, 60),   29)),   # 0-9%   green
            (20,  ((34, 130, 55),   29)),   # 10-19%
            (30,  ((34, 140, 50),   35)),   # 20-29%
            (40,  ((50, 150, 40),   35)),   # 30-39%
            (50,  ((70, 160, 30),   70)),   # 40-49%
            (60,  ((130, 140, 0),   142)),  # 50-59% yellow-green
            (70,  ((160, 130, 0),   178)),  # 60-69% olive/yellow
            (80,  ((180, 100, 0),   172)),  # 70-79% orange
            (90,  ((200, 60, 0),    166)),  # 80-89% red-orange
            (101, ((210, 30, 30),   160)),  # 90-100% red
        ],
    },
}

# =============================================================================
# COLOR SUPPORT DETECTION
# =============================================================================

def supports_truecolor():
    """Detect if terminal supports 24-bit true color"""
    colorterm = os.environ.get('COLORTERM', '').lower()
    return colorterm in ('truecolor', '24bit')

TRUECOLOR = supports_truecolor()

# =============================================================================
# ANSI ESCAPE HELPERS
# =============================================================================

RESET = '\033[0m'
BOLD = '\033[1m'

def _color(rgb, fallback_256, is_bg=False):
    """Generate ANSI color code with truecolor/256 fallback"""
    prefix = 48 if is_bg else 38
    if TRUECOLOR and rgb is not None:
        return f'\033[{prefix};2;{rgb[0]};{rgb[1]};{rgb[2]}m'
    else:
        return f'\033[{prefix};5;{fallback_256}m'

def fg_themed(color_tuple):
    """Foreground color from theme tuple ((rgb, _), fallback) or ((rgb, fallback), _)"""
    if isinstance(color_tuple[0], tuple):
        rgb, fallback = color_tuple[0]
        if fallback is None:
            fallback = color_tuple[1]
    else:
        rgb, fallback = color_tuple
    return _color(rgb, fallback, is_bg=False)

def bg_themed(color_tuple):
    """Background color from theme tuple ((rgb, fallback), _)"""
    rgb, fallback = color_tuple[0]
    return _color(rgb, fallback, is_bg=True)

def fg_gradient(rgb, fallback_256):
    """Foreground from gradient tuple"""
    return _color(rgb, fallback_256, is_bg=False)

def fg_empty():
    """Foreground for empty bar portion"""
    theme = THEMES[THEME]
    rgb, fallback = theme['bar_empty']
    return _color(rgb, fallback, is_bg=False)

# =============================================================================
# THEME-AWARE COLOR FUNCTIONS
# =============================================================================

def get_colors_for_percentage(pct):
    """Return (rgb, fallback_256) for progress bar fill at given percentage"""
    theme = THEMES[THEME]
    for threshold, color in theme['gradient']:
        if pct < threshold:
            return color
    return theme['gradient'][-1][1]

def get_model_colors(model):
    """Return (bg_code, fg_code) for model badge"""
    theme = THEMES[THEME]
    if 'Sonnet' in model:
        key = 'model_sonnet'
    elif 'Opus' in model:
        key = 'model_opus'
    elif 'Haiku' in model:
        key = 'model_haiku'
    else:
        key = 'model_default'

    bg_tuple, fg_tuple = theme[key]
    bg_code = _color(bg_tuple[0], bg_tuple[1], is_bg=True)
    fg_code = _color(fg_tuple[0], fg_tuple[1], is_bg=False)
    return bg_code + BOLD + fg_code

def text_color(key):
    """Get text color by key: 'percent', 'numbers', 'cwd', 'git'"""
    theme = THEMES[THEME]
    color_tuple = theme[f'text_{key}']
    return fg_themed(color_tuple)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def center_text(text, min_width=12):
    """Center text with 1-char padding on each side, minimum 12 chars wide"""
    width = max(min_width, len(text) + 2)
    padding = (width - len(text)) // 2
    right_padding = width - len(text) - padding
    return ' ' * padding + text + ' ' * right_padding

def get_git_branch(cwd):
    """Get current git branch, or None"""
    try:
        result = subprocess.run(
            ['git', '-C', cwd, 'branch', '--show-current'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None

def get_cwd_suffix(cwd):
    """Format cwd and git branch for display"""
    if not cwd:
        return ''

    # Shorten home directory to ~
    home = os.path.expanduser('~')
    if cwd.startswith(home):
        cwd_short = '~' + cwd[len(home):]
    else:
        cwd_short = cwd

    suffix = f'   {text_color("cwd")}{cwd_short}'

    git_branch = get_git_branch(cwd)
    if git_branch:
        suffix += f'   {BOLD}{text_color("git")}[{git_branch}]'

    return suffix

# =============================================================================
# MAIN STATUS LINE BUILDER
# =============================================================================

 "**********"d "**********"e "**********"f "**********"  "**********"b "**********"u "**********"i "**********"l "**********"d "**********"_ "**********"p "**********"r "**********"o "**********"g "**********"r "**********"e "**********"s "**********"s "**********"_ "**********"b "**********"a "**********"r "**********"( "**********"p "**********"c "**********"t "**********", "**********"  "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"c "**********"w "**********"d "**********", "**********"  "**********"t "**********"o "**********"t "**********"a "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********", "**********"  "**********"c "**********"o "**********"n "**********"t "**********"e "**********"x "**********"t "**********"_ "**********"l "**********"i "**********"m "**********"i "**********"t "**********") "**********": "**********"
    """Build the full status line string"""
    bar_length = int(100 * BAR_RATIO + 0.5)
    filled = int(pct * BAR_RATIO + 0.5)
    empty = bar_length - filled

    bar_rgb, bar_256 = get_colors_for_percentage(pct)
    model_color = get_model_colors(model)

    parts = [
        model_color + center_text(model) + RESET,
        fg_gradient(bar_rgb, bar_256) + '█' * filled,
        fg_empty() + '█' * empty,
        RESET + text_color('percent'),
        f' {pct}%',
        text_color('numbers'),
        f' ({total_tokens // 1000}k/{context_limit // 1000}k)',
        get_cwd_suffix(cwd),
        RESET
    ]

    return ''.join(parts)

def build_na_line(model, cwd):
    """Build status line when no usage data available"""
    model_color = get_model_colors(model)
    suffix = get_cwd_suffix(cwd)
    return f'{model_color}{center_text(model)}{RESET} {text_color("percent")}context size N/A{suffix}{RESET}'

# =============================================================================
# DEMO MODE
# =============================================================================

def show_scale_demo(mode='animate'):
    """Demo mode to show color gradient"""
    import time

    def show_bar(pct):
        bar_length = int(100 * BAR_RATIO + 0.5)
        filled = int(pct * BAR_RATIO + 0.5)
        empty = bar_length - filled
        bar_rgb, bar_256 = get_colors_for_percentage(pct)
        bar = fg_gradient(bar_rgb, bar_256) + '█' * filled + fg_empty() + '█' * empty + RESET
        return bar

    if mode == 'animate':
        try:
            while True:
                for pct in range(101):
                    print(f'\r{pct:3d}%: {show_bar(pct)}', end='', flush=True)
                    time.sleep(0.1)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()
    elif mode in ('min', 'max', 'mid'):
        ranges = [(0,9), (10,19), (20,29), (30,39), (40,49),
                  (50,59), (60,69), (70,79), (80,89), (90,100)]
        print(f'Color Scale Demo ({mode} value):')
        print()
        for lo, hi in ranges:
            pct = lo if mode == 'min' else hi if mode == 'max' else (lo + hi) // 2
            print(f'{lo:3d}-{hi:3d}%: {show_bar(pct)}')
    else:
        print(f"Error: Invalid mode '{mode}'. Use: min, max, mid, or animate")
        sys.exit(1)

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Handle --show-scale demo mode
    if len(sys.argv) > 1 and sys.argv[1] == '--show-scale':
        show_scale_demo(sys.argv[2] if len(sys.argv) > 2 else 'animate')
        return

    # Read and parse JSON input
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        print('statusline: invalid JSON input', file=sys.stderr)
        return

    model = data.get('model', {}).get('display_name', 'Claude')
    cwd = data.get('cwd', '')

    # Get context window info
    context_window = data.get('context_window', {})
    context_limit = context_window.get('context_window_size', 200000)
    current_usage = context_window.get('current_usage')

    # Calculate total tokens (input + cache, not output)
    if current_usage:
        total_tokens = "**********"
            current_usage.get('input_tokens', 0) +
            current_usage.get('cache_creation_input_tokens', 0) +
            current_usage.get('cache_read_input_tokens', 0)
        )
    else:
        total_tokens = "**********"

    # Handle no usage data yet
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"t "**********"a "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        print(build_na_line(model, cwd))
        return

    # Calculate percentage (capped at 100)
    pct = "**********"

    print(build_progress_bar(pct, model, cwd, total_tokens, context_limit))

if __name__ == '__main__':
    main()
