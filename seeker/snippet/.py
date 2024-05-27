#date: 2024-05-27T16:48:15Z
#url: https://api.github.com/gists/99c617d2c10b8b70fa9f7abd82550d3a
#owner: https://api.github.com/users/jdgr29

class RushHour:
    def __init__(self, board, exit_row=2):
        self.board = board
        self.exit_row = exit_row

    def display_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    def move_vehicle(self, vehicle, direction, steps):
        if direction not in ('up', 'down', 'left', 'right'):
            print("Invalid direction!")
            return False

        vehicle_coords = self.find_vehicle(vehicle)
        if not vehicle_coords:
            print("Vehicle not found!")
            return False

        if not self.can_move(vehicle_coords, direction, steps):
            print("Move not possible!")
            return False

        self.apply_move(vehicle_coords, vehicle, direction, steps)
        return True

    def find_vehicle(self, vehicle):
        vehicle_coords = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                if self.board[r][c] == vehicle:
                    vehicle_coords.append((r, c))
        return vehicle_coords

    def can_move(self, coords, direction, steps):
        for r, c in coords:
            if direction == 'up' and (r - steps < 0 or self.board[r - steps][c] not in ('.', self.board[r][c])):
                return False
            if direction == 'down' and (r + steps >= len(self.board) or self.board[r + steps][c] not in ('.', self.board[r][c])):
                return False
            if direction == 'left' and (c - steps < 0 or self.board[r][c - steps] not in ('.', self.board[r][c])):
                return False
            if direction == 'right' and (c + steps >= len(self.board[r]) or self.board[r][c + steps] not in ('.', self.board[r][c])):
                return False
        return True

    def apply_move(self, coords, vehicle, direction, steps):
        # Clear the vehicle's current position
        for r, c in coords:
            self.board[r][c] = '.'

        # Move the vehicle to the new position
        new_coords = []
        for r, c in coords:
            if direction == 'up':
                new_coords.append((r - steps, c))
            elif direction == 'down':
                new_coords.append((r + steps, c))
            elif direction == 'left':
                new_coords.append((r, c - steps))
            elif direction == 'right':
                new_coords.append((r, c + steps))

        for r, c in new_coords:
            self.board[r][c] = vehicle

    def check_win(self):
        for c in range(len(self.board[self.exit_row])):
            if self.board[self.exit_row][c] == 'R':
                if c == len(self.board[self.exit_row]) - 1:
                    return True
        return False


def main():
    board = [
        ['G', 'G', '.', '.', '.', '.'],
        ['B', '.', '.', 'O', 'O', 'O'],
        ['B', '.', '.', 'R', 'R', '.'],
        ['B', '.', '.', '.', '.', '.'],
        ['.', 'Y', 'Y', '.', 'P', 'P'],
        ['.', '.', '.', '.', '.', '.']
    ]

    game = RushHour(board)
    game.display_board()

    while not game.check_win():
        vehicle = input(
            "Enter vehicle to move (e.g., 'R', 'G'): ").strip().upper()
        direction = input(
            "Enter direction (up, down, left, right): ").strip().lower()
        steps = int(input("Enter number of steps to move: ").strip())

        if game.move_vehicle(vehicle, direction, steps):
            game.display_board()
        else:
            print("Invalid move, try again.")

    print("Congratulations! You've won the game.")


if __name__ == "__main__":
    main()
