#date: 2024-03-12T16:47:45Z
#url: https://api.github.com/gists/27623af37c01308fcdfdd407ca08fa42
#owner: https://api.github.com/users/george-cosma

import curses

def fit_text_in_box(cols, text):
    # Calculate the number of columns required
    columns = cols
    text_lines = []

    words = text.split()
    text_lines = []
    current_line = ''

    for word in words:
        # If adding the next word to the current line would exceed the maximum length,
        # start a new line
        if len(current_line) + len(word) + 1 > columns:
            text_lines.append(current_line)
            current_line = word
        else:
            # If the current line is not empty, add a space before the next word
            if current_line:
                current_line += ' ' + word
            else:
                current_line = word

    # Add the last line to the list of lines
    text_lines.append(current_line)

    # Make sure all lines have the same length by adding spaces to the end if necessary
    text_lines = [line.ljust(columns) for line in text_lines]


    output = ""

    # Print the box with the text
    output += f".{'-' * (columns + 2)}.\n"
    for line in text_lines:
        output += f"| {line} |\n"
    output += f"'{'-' * (columns + 2)}'\n"

    return output

def print_clear(stdscr, text):
    stdscr.clear()
    stdscr.addstr(text)
    stdscr.refresh()


def main(stdscr):
    # Initialize curses
    stdscr.clear()
    curses.curs_set(0)
    # stdscr.nodelay(True)
    curses.echo()

    stdscr.addstr("Enter the text: ")
    stdscr.refresh()
    text = stdscr.getstr().decode()
    cols = min(30, len(text) + 2)

    curses.noecho()


    # Display the initial box
    print_clear(stdscr, fit_text_in_box(cols, text))

    while True:
        # Wait for a key press
        c = stdscr.getch()

        if c == curses.KEY_RIGHT:
            # Increase the number of columns
            # text = text.ljust(len(text) + rows)
            cols = cols + 1
            print_clear(stdscr, fit_text_in_box(cols, text))
        elif c == curses.KEY_LEFT:
            # Decrease the number of columns
            # text2 = text[:-rows]
            # if len(text2) < original_len:
            #     continue

            # text = text2
            cols = max(3, cols - 1)
            print_clear(stdscr, fit_text_in_box(cols, text))
        elif c == ord('q'):
            # Quit the program
            break


if __name__ == "__main__":
    curses.wrapper(main)
