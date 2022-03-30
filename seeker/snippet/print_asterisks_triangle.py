#date: 2022-03-30T17:02:17Z
#url: https://api.github.com/gists/ce1fbd493a5cf0b8287f8d1072fe8be4
#owner: https://api.github.com/users/jaimeHMol

def print_asterisks_triangle(size):
    """
    Print a triangle of asterisks receiving the size of the base.
    """
    for line in range(1, size+1):
        line_template = list(" " * size * 2)
        for asterisk in range(0, line):
            starting_point = size - line
            x_offset = asterisk * 2
            current_position = starting_point + x_offset
            line_template[current_position] = "*"
        print("".join(line_template))
