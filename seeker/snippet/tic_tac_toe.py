#date: 2023-11-02T16:50:48Z
#url: https://api.github.com/gists/f74e18d49cc99bd434aae752c3584200
#owner: https://api.github.com/users/Zai-Kun

import os

def get_all_max_numbers(input_list):
    if not input_list:
        return []

    max_value = max(input_list)[0]
    max_numbers = [num for num in input_list if num[0] == max_value]

    return max_numbers

def get_all_min_numbers(input_list):
    if not input_list:
        return []

    min_value = min(input_list)[0]
    min_numbers = [num for num in input_list if num[0] == min_value]

    return min_numbers

class Board:
    def __init__(self):
        self.board_skeleton = """
{} | {} | {}
---------
{} | {} | {}
---------
{} | {} | {} 
"""
        self.board = [i for i in range(9)]
        self.player = "X"
        self.computer = "O"
        self.values = {self.player: -1, self.computer: 1}

    def print_board(self):
        print(self.board_skeleton.format(*self.board))

    def check_winner(self):
        wins = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        )
        for player in (self.player, self.computer):
            for a, b, c in wins:
                if self.board[a] == self.board[b] == self.board[c] == player:
                    return player
        return None

    def get_legal_moves(self):
        return [cell for cell in self.board if cell != self.player and cell != self.computer]

    def game_over(self):
        return not self.get_legal_moves()

    def count_moves(self):
        moves = {self.player: 0, self.computer: 0}
        for cell in self.board:
            if cell == self.player or cell == self.computer:
                moves[cell] += 1
        return moves

    def turn_to_move(self):
        move_counts = self.count_moves()
        count_player = move_counts[self.player]
        count_computer = move_counts[self.computer]

        if count_player == count_computer:
            return self.player
        elif count_player > count_computer:
            return self.computer
        else:
            return self.player
    def evaluate_game_state(self):
        winner = self.check_winner()
        if winner:
            return self.values[winner]
        elif self.game_over():
            return 0
        else:
            return None

    def make_move(self, move):
        turn_to_move = self.turn_to_move()
        self.board[move] = turn_to_move
    
    def undo_move(self, move):
        self.board[move] = move
    
    def minimax(self, maximizing, depth=0):
        game_state = self.evaluate_game_state()
        if game_state != None:
            return game_state, depth
        
        legal_moves = self.get_legal_moves()
        best_score = [float("-inf") if maximizing else float("inf")]


        for legal_move in legal_moves:
            self.make_move(legal_move)
            if maximizing:
                best_score = max(best_score, self.minimax(maximizing=False, depth=depth + 1), key=lambda x:x[0])
            else:
                best_score = min(best_score, self.minimax(maximizing=True, depth=depth + 1), key=lambda x:x[0])
            self.undo_move(legal_move)

        return best_score

    def get_best_move(self):
        maximizing = self.turn_to_move() == self.computer
        get_all_max_or_min_numbers = get_all_max_numbers if maximizing else get_all_min_numbers
        
        legal_moves = self.get_legal_moves()
        total_scores = []

        for legal_move in legal_moves:
            self.make_move(legal_move)
            move_score = self.minimax(maximizing=not maximizing)
            self.undo_move(legal_move)
            total_scores.append([move_score, legal_move])

        best_score = min(get_all_max_or_min_numbers([score[0] for score in total_scores]), key=lambda x:x[-1])
        for score in total_scores:
            if score[0] == best_score:
                return score[-1]

def start_game(board):
    os.system("clear")
    while True:
        board.print_board()
        player_move = int(input("\nEnter here: "))
        os.system("clear")
        
        legal_moves = board.get_legal_moves()
        if player_move in legal_moves:
            board.make_move(player_move)
        else:
            print("Not a legal move. Try again.")

        winner = board.check_winner()
        if winner or board.game_over():
            break

        com_move = board.get_best_move()
        board.make_move(com_move)

        winner = board.check_winner()
        if winner or board.game_over():
            break

    if not winner:
        print("Draw")
    elif winner == board.computer:
        print("Computer won, sucker.")
    else:
        print("Tch....")

if __name__ == "__main__":
    board = Board()
    start_game(board)