#date: 2025-04-16T16:55:38Z
#url: https://api.github.com/gists/7546735549891a344e4d2451adc56f13
#owner: https://api.github.com/users/mousberg

import math
import random
from typing import List, Optional, Tuple

# Constants for players
PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = " "

def print_board(board: List[str]) -> None:
    """Prints the Tic-Tac-Toe board."""
    print("\\n-------------")
    for i in range(0, 9, 3):
        print(f"| {board[i]} | {board[i+1]} | {board[i+2]} |")
        print("-------------")
    print()

def check_winner(board: List[str]) -> Optional[str]:
    """
    Checks if there is a winner.
    Returns the winner ('X' or 'O') or None if no winner yet.
    """
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
        (0, 4, 8), (2, 4, 6)             # Diagonals
    ]
    for a, b, c in win_conditions:
        if board[a] == board[b] == board[c] and board[a] != EMPTY:
            return board[a]
    return None

def is_board_full(board: List[str]) -> bool:
    """Checks if the board is full."""
    return EMPTY not in board

def get_available_moves(board: List[str]) -> List[int]:
    """Returns a list of indices for available moves."""
    return [i for i, spot in enumerate(board) if spot == EMPTY]

def minimax(board: List[str], depth: int, is_maximizing: bool, alpha: float, beta: float) -> int:
    """
    Minimax algorithm with alpha-beta pruning to determine the best score.
    """
    winner = check_winner(board)
    if winner == PLAYER_O:  # AI (O) wins
        return 10 - depth
    if winner == PLAYER_X:  # Human (X) wins
        return depth - 10
    if is_board_full(board):  # Draw
        return 0

    if is_maximizing:
        max_eval = -math.inf
        for move in get_available_moves(board):
            board[move] = PLAYER_O
            evaluation = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY # Backtrack
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break # Beta cut-off
        return max_eval
    else: # Minimizing player (Human)
        min_eval = math.inf
        for move in get_available_moves(board):
            board[move] = PLAYER_X
            evaluation = minimax(board, depth + 1, True, alpha, beta)
            board[move] = EMPTY # Backtrack
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break # Alpha cut-off
        return min_eval

def find_best_move(board: List[str]) -> int:
    """
    Finds the best move for the AI (Player O) using Minimax.
    """
    best_score = -math.inf
    best_move = -1
    available_moves = get_available_moves(board)

    # Add a bit of randomness for opening moves if board is empty or nearly empty
    if len(available_moves) >= 8:
       return random.choice(available_moves)

    for move in available_moves:
        board[move] = PLAYER_O
        # Start minimax for the minimizing player (human's turn after AI moves)
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move] = EMPTY  # Backtrack
        if score > best_score:
            best_score = score
            best_move = move
        # Introduce slight randomness among equally good moves
        elif score == best_score and random.choice([True, False]):
             best_move = move


    # Fallback if no best move found (shouldn't happen in standard play)
    if best_move == -1 and available_moves:
        return random.choice(available_moves)

    return best_move


def get_player_move(board: List[str]) -> int:
    """Gets the human player's move."""
    while True:
        try:
            move = int(input(f"Player {PLAYER_X}, enter your move (1-9): ")) - 1
            if 0 <= move <= 8:
                if board[move] == EMPTY:
                    return move
                else:
                    print("This spot is already taken. Try again.")
            else:
                print("Invalid input. Please enter a number between 1 and 9.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def play_game() -> None:
    """Main game loop."""
    board = [EMPTY] * 9
    current_player = PLAYER_X  # Human starts

    print("Welcome to Tic-Tac-Toe!")
    print("You are Player X. The AI is Player O.")
    print("Board positions are numbered 1-9, left-to-right, top-to-bottom.")

    while True:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"Player {winner} wins!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        if current_player == PLAYER_X:
            move = get_player_move(board)
            board[move] = PLAYER_X
            current_player = PLAYER_O
        else: # AI's turn
            print("AI (Player O) is thinking...")
            move = find_best_move(board)
            if move != -1:
                board[move] = PLAYER_O
                print(f"AI chose spot {move + 1}")
            else:
                 # Should only happen if board is full, but handled defensively
                 print("AI cannot find a move.")

            current_player = PLAYER_X

if __name__ == "__main__":
    play_game() 