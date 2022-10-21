#date: 2022-10-21T17:14:50Z
#url: https://api.github.com/gists/69100b18e79048853bc05aa108cda2d6
#owner: https://api.github.com/users/codewitgabi

import sys

from random import randrange

from time import sleep

sys.setrecursionlimit(40)

board = {"00": "", "01": "", "02": "", "10": "", "11": "", "12": "", "20": "", "21": "", "22": ""}

COUNTER = 0

MAX_PLAY = sys.getrecursionlimit() * 5.4

computer_score = 0

human_score = 0

	def draw_board():

	print(board["00"] + "|" + board["01"] + "|" +board["02"])

	print("-+-+-")

	print(board["10"] + "|" + board["11"] + "|" +board["12"])

	print("-+-+-")

	print(board["20"] + "|" + board["21"] + "|" +board["22"])

	

	

def human_move():

	try:

		move = input("Human(X): ")

		if board[move] == "":

			board[move] = "X"

			draw_board()

			

		else:

			print("Invalid Input!!!")

			human_move()

		

			

	except KeyError:

		print("Invalid Input!!!")

		human_move()

		

	

def computer_move():

	while True:

		row = randrange(3)

		column = randrange(3)

		

		if board[str(row) + str(column)] == "":

			board[str(row) + str(column)] = "O"

			print("Computer: thinking...")

			sleep(1.5)

			draw_board()

			break

		

	

def check_for_winner():

	# rows

	for i in range(3):

		if board[f"{i}0"] == board[f"{i}1"] and board[f"{i}0"] == board[f"{i}2"]:

			return board[f"{i}0"]

			

	# columns

	for i in range(3):

		if board[f"0{i}"] == board[f"1{i}"] and board[f"0{i}"] == board[f"2{i}"]:

			return board[f"0{i}"]

			

	# diagonals

	if board["00"] == board["11"] and board["00"] == board["22"]:

		return board["00"]

		

	if board["02"] == board["11"] and board["02"] == board["20"]:

			return board["02"]

	

def main():

	moves = 9

	global COUNTER

	global board

	global human_score

	global computer_score

	winner = ""

	

	if COUNTER < MAX_PLAY:

		while moves > 0:

			human_move()

			winner = check_for_winner()

			moves -= 1

		

			if winner is not None and winner != "":

				print("Human WON!!!")

				moves = 9

				board = {"00": "", "01": "", "02": "", "10": "", "11": "", "12": "", "20": "", "21": "", "22": ""}

				human_score += 1

				print(f"PLAYER: {human_score}      COMPUTER: {computer_score}")

				main()

				

			if moves == 0:

				break

			

			computer_move()

			winner = check_for_winner()

			moves -= 1

			

			if winner is not None and winner != "":

				print("Compter WON!!!")

				moves = 9

				board = {"00": "", "01": "", "02": "", "10": "", "11": "", "12": "", "20": "", "21": "", "22": ""}

				computer_score += 1

				print(f"PLAYER: {human_score}      COMPUTER: {computer_score}")

				main()

				

			COUNTER += 1

					

		print("It is a draw")

		board = {"00": "", "01": "", "02": "", "10": "", "11": "", "12": "", "20": "", "21": "", "22": ""}

		main()

	else:

		return

print("====================================")

print("            TIC TAC TOE")

print("""

HOW TO PLAY:

	

	00|01|02

	--+--+--

	10|11|12

	--+--+--

	20|21|22

	

	Each number signifies the row

	and column to input your value

	Dont't worry, the game is in easy mode'

	""")

print("====================================")

		

main()

	