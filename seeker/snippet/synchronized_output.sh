#date: 2026-01-14T17:06:56Z
#url: https://api.github.com/gists/12c017ecf54e1310b7f7f712bbc044a9
#owner: https://api.github.com/users/TyOverby

#!/usr/bin/env bash

# Synchronized output.  
# Use these escape codes to tell the terminal emulator when a "frame" starts and ends so that
# it can buffer stdout and only display complete terminal contents - no screen tearing! 
# Docs: https://github.com/contour-terminal/vt-extensions/blob/master/synchronized-output.md

# ANSI escape codes
BSU='\e[?2026h'  # Begin Synchronized Update
ESU='\e[?2026l'  # End Synchronized Update
CLEAR='\e[2J'    # Clear screen
HOME='\e[H'      # Move cursor to home position

# Helper function to render a frame
render_frame() {
    local frame_num=$1
    local label=$2
    
    printf "${CLEAR}${HOME}"
    echo "╔════════════════════════════════╗"
    echo "║   Frame $frame_num $label    ║"
    echo "╠════════════════════════════════╣"
    sleep 0.05
    echo "║ Line 1: Processing...          ║"
    sleep 0.05
    echo "║ Line 2: Loading data...        ║"
    sleep 0.05
    echo "║ Line 3: Computing results...   ║"
    sleep 0.05
    echo "║ Line 4: Almost done...         ║"
    sleep 0.05
    echo "╚════════════════════════════════╝"
}

echo "=== Demonstration of Synchronized Output ==="
echo ""
echo "First, let's see the BAD behavior (with tearing):"
echo "Press Enter to continue..."
read

for i in {1..5}; do
    render_frame "$i" "(WITHOUT sync)   "
    sleep 0.1
done

echo ""
echo "Notice the tearing/flicker as lines appeared one by one?"
echo ""
echo "Now with synchronized output (smooth updates):"
echo "Press Enter to continue..."
read

for i in {1..5}; do
    printf "${BSU}"  # Begin synchronized update
    render_frame "$i" "(WITH sync)      "
    printf "${ESU}"  # End synchronized update - render all at once!
    sleep 0.1
done

echo ""
echo "Much smoother! Each frame appeared all at once."
