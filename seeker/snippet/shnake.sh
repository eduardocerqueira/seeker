#date: 2023-07-24T16:39:47Z
#url: https://api.github.com/gists/aface4ca59be5882ab0c4ff2441352b8
#owner: https://api.github.com/users/mr-ema

#!/bin/sh

# -----------------------------------------------------------------------------
# Zero-Clause BSD
#
# Permission to use, copy, modify, and/or distribute this software for
# any purpose with or without fee is hereby granted.
# 
# THE SOFTWARE IS PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
# AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# -----------------------------------------------------------------------------

cleanup() {
    tput cnorm
}

trap cleanup EXIT
tput civis

#VERSION="1.0"
#PROGRAM="shnake"

#FPS=60
#FRAME_DURATION=$($bc -I <<< "1 / $FPS")

SCREEN_WIDTH=$(tput cols)
SCREEN_HEIGHT=$(tput lines)
GRID_WIDTH=$((SCREEN_WIDTH - 1))
GRID_HEIGHT=$((SCREEN_HEIGHT - 1))

MATRIX_CHAR="."
HEAD_CHAR="O"
TAIL_CHAR="o"
FRUIT_CHAR="F"

LEFT_KEY="h"
RIGHT_KEY="l"
UP_KEY="k"
DOWN_KEY="j"

# Variables
snake_x=0
snake_y=0
fruit_x=0
fruit_y=0
tail_x=0
tail_y=0

# since posix does not specify arrays,the tail will be represent as a string
# like: [xy: "x0y1 x1y0 x2y3 x5y5"]
snake_tail_xy=""
direction="RIGHT"
matrix=""
pressed_key=""
#score=0

replace_char() {
        str=$1
        idx=$2
        new_char=$3

        if [ "$idx" -ge 0 ] && [ "$idx" -lt ${#str} ]; then
                new_str=$(echo "$str" | sed "s/./${new_char}/$((idx + 1))")
                echo "$new_str"
        else
                echo "$str"
        fi
}

update_tail() {
        # remove last segment and then update tail
        snake_tail_xy=$(echo "$snake_tail_xy" |  awk -F 'x' 'NF>1{sub(/x[^x]*$/,"")}1')
        snake_tail_xy="x${snake_x}y${snake_y} $snake_tail_xy"
}

draw_snake() {
        # Draw snake tail
        for tail_xy in $snake_tail_xy; do
                tail_x=$(echo "$tail_xy" | cut -d'x' -f2 | cut -d'y' -f1)
                tail_y=$(echo "$tail_xy" | cut -d'y' -f2 | cut -d'x' -f1)

                # We do 'GRID_WIDTH + 2' because of new line excape '\n'
                tail_idx=$((tail_y * (GRID_WIDTH + 2) + tail_x))
                matrix=$(replace_char "$matrix" "$tail_idx" "$TAIL_CHAR")
        done

        head_idx=$((snake_y * (GRID_WIDTH + 2) + snake_x))
        matrix=$(replace_char "$matrix" "$head_idx" "$HEAD_CHAR")
}

draw_game() {
        # Draw matrix one time
        if [ -z "$matrix" ]; then
                for _ in $(seq 0 $((GRID_HEIGHT - 1))); do
                        for _ in $(seq 0 $((GRID_WIDTH - 1))); do
                                matrix="${matrix}$MATRIX_CHAR"
                        done
                        matrix="${matrix}\n"
                done
        else
                # remove the last tail segment to simulate movement
                last_segment_idx=$((tail_y * (GRID_WIDTH + 2) + tail_x))
                matrix=$(replace_char "$matrix" "$last_segment_idx" "$MATRIX_CHAR")
        fi

        # Draw fruit
        fruit_idx=$((fruit_y * (GRID_WIDTH + 2) + fruit_x))
        matrix=$(replace_char "$matrix" "$fruit_idx" "$FRUIT_CHAR")

        draw_snake
        printf -- "$matrix"
}

move_snake() {
        pressed_key=$1

        case "$pressed_key" in
                "$LEFT_KEY") direction="LEFT";;
                "$UP_KEY") direction="UP";;
                "$RIGHT_KEY") direction="RIGHT";;
                "$DOWN_KEY") direction="DOWN";;
        esac

        case "${direction}" in
                "UP") snake_y=$((snake_y - 1));;
                "DOWN") snake_y=$((snake_y + 1));;
                "LEFT") snake_x=$((snake_x - 1));;
                "RIGHT") snake_x=$((snake_x + 1));;
        esac
}

generate_fruit() {
        if [ -n "$RANDOM" ]; then
                fruit_x=$((RANDOM % GRID_WIDTH))
                fruit_y=$((RANDOM % GRID_HEIGHT))
        else
                current_time=$(date +%s%3N)

                fruit_x=$((current_time % GRID_WIDTH))
                fruit_y=$((current_time % GRID_HEIGHT))
        fi
}

check_collition() {
        if [ "$snake_x" -lt 0 ] || [ "$snake_x" -ge "$GRID_WIDTH" ] || [ "$snake_y" -lt 0 ] || [ "$snake_y" -ge "$GRID_HEIGHT" ]; then
                echo "GAME OVER"
                exit 0
        elif [ "$snake_x" -eq "$fruit_x" ] && [ "$snake_y" -eq "$fruit_y" ]; then
                snake_tail_xy="x${snake_x}y${snake_y} $snake_tail_xy"
                generate_fruit
        fi
}

non_blocking_read() {
        time=$1
        varname=$2

        if [ -n "$ZSH_VERSION" ]; then
                read -k1 -s -t $time $varname
        else
                read -r -n 1 -t $time $varname
        fi
}

# Main loop
generate_fruit # Spawn the fruit in a random position
while true; do
        draw_game

        check_collition

        non_blocking_read 0.2 pressed_key
        move_snake "$pressed_key"
        update_tail
done
