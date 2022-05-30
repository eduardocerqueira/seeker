#date: 2022-05-30T17:03:09Z
#url: https://api.github.com/gists/adc33dd1dd6d0a9db094b1afe9beda2a
#owner: https://api.github.com/users/codingwarrior21

def update_coordinate(direction,coordinates):
    coordinates.pop(-1)
    if direction == "north":
        column_position = coordinates[0][0]
        row_position = coordinates[0][1] - 1
    elif direction == "south":
        column_position = coordinates[0][0]
        row_position = coordinates[0][1] + 1
    elif direction == "east":
        column_position = coordinates[0][0] + 1
        row_position = coordinates[0][1]
    elif direction == "west":
        column_position = coordinates[0][0] - 1
        row_position = coordinates[0][1]  
    return column_position, row_position 