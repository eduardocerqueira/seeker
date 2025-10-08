#date: 2025-10-08T16:46:17Z
#url: https://api.github.com/gists/0664fca09d758b2d8fb6fe6c73d610ad
#owner: https://api.github.com/users/BennettPompiM7

# open_localhost.py
import subprocess
import sys

URL = "http://localhost:3000"

osascript = f'''
set targetURL to "{URL}"

try
    tell application "Google Chrome"
        -- Launch Chrome if needed
        if not application "Google Chrome" is running then
            activate
            delay 0.5
        end if

        -- Ensure at least one window exists (some profiles launch w/out a window)
        if (count of windows) = 0 then
            make new window
            delay 0.1
        end if

        -- Look for an existing localhost:3000 tab
        repeat with w in windows
            set tabIndex to 0
            repeat with t in tabs of w
                set tabIndex to tabIndex + 1
                set u to (URL of t)
                if u starts with targetURL or u starts with "https://localhost:3000" then
                    set active tab index of w to tabIndex
                    set index of w to 1
                    activate
                    return
                end if
            end repeat
        end repeat

        -- Otherwise open a new tab in the front window
        tell window 1 to make new tab with properties {{URL:targetURL}}
        activate
    end tell
on error errMsg number errNum
    return "AS_ERROR|" & errNum & "|" & errMsg
end try
'''

proc = subprocess.run(
    ["osascript", "-e", osascript],
    capture_output=True,
    text=True
)

out = (proc.stdout or "").strip()
if proc.returncode != 0 or out.startswith("AS_ERROR|"):
    # Show a readable error and fall back to a simple open
    err = out if out else (proc.stderr or "").strip()
    print(f"AppleScript failed: {err}", file=sys.stderr)
    # Fallback: just open/focus Chrome with the URL
    subprocess.run(["open", "-ga", "Google Chrome", URL], check=False)