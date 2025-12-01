#date: 2025-12-01T17:07:55Z
#url: https://api.github.com/gists/1d76922db7ad35075088267a57b82c9c
#owner: https://api.github.com/users/joshua-fox-fe

#!/usr/bin/env python3

"""Padlock simulation.

Padlock has 100 positions (0-99). We start at 50 and apply a series of
left (L) or right (R) rotations given as e.g. "L30" meaning rotate 30 left.
We track how many times the dial points exactly at 0 AFTER a move.
"""

INPUT_DATA = ["L68", "L30", "R48", "L5", "R60", "L55", "L1", "L99", "R14", "L82"]

def cut_char_then_int(token: "**********":
    """Strip first character (direction) and return remaining integer."""
    return int(token[1: "**********"

def update_pointed_count(position: int, pointed_count: int) -> int:
    """Increment pointed_count if position is 0."""
    if position == 0:
        pointed_count += 1
    return pointed_count

def parse_and_move(start_position: int, start_pointed_count: int, moves) -> tuple[int, int]:
    position = start_position
    pointed_count = start_pointed_count
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"m "**********"o "**********"v "**********"e "**********"s "**********": "**********"
        direction = "**********"
        clicks = "**********"
        if direction == "L":
            position = (position - clicks) % 100
        elif direction == "R":
            position = (position + clicks) % 100
        else:
            print(f"ERROR: "**********"
            continue
        pointed_count = update_pointed_count(position, pointed_count)
        print(f"DEBUG: "**********": position={position} pointed_count={pointed_count}")
    return position, pointed_count

def main():
    start_position = 50
    pointed_count = 0
    final_position, final_pointed = parse_and_move(start_position, pointed_count, INPUT_DATA)
    print(f"Final position: {final_position}. Times pointed to 0: {final_pointed}")

if __name__ == "__main__":
    main()
