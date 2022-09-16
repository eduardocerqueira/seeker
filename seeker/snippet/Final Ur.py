#date: 2022-09-16T17:12:04Z
#url: https://api.github.com/gists/24a12c82ce2a8814d08228fe09b5fb62
#owner: https://api.github.com/users/wolfboyft

# PyUr — copyleft 2017 — is by
# Henry "wolfboyft" Fleminger Thomson, and
# licensed under the GNU GPL 3.
# Tested with Python 3.5

from random import randint
import curses

white = "White" # Constant
black = "Black" # Constant
rosettas = [4, 8, 14] # Constant
gamex = 47 # Constant
gamey = 7 # Constant
game = [ # Constant
"┌────────────────────────┐",
"│┌─────────────┬┬───────┐│",
"││@←← ←← ←← ←← ││ ←←@←← ││",
"││↓┌───────────┘└─────┐↑││",
"││↓└──────────────────┘↑││",
"││ →→ →→ →→@→→ →→ →→ →→ ││",
"││↑┌──────────────────┐↓││",
"││↑└───────────┐┌─────┘↓││",
"││@←← ←← ←← ←← ││ ←←@←← ││",
"│└─────────────┴┴───────┘│",
"├────────────────────────┤",
"│                        │", # Status window.
"├────────────────────── ─┤",
"│                        │", # Message window.
"└────────────────────────┘"]

help = [ # Constant
[
"┌────────────────────────┐",
"│1: Navigation & contents│",
"│                        │",
"│Press 1 to 8 to see a   │",
"│page or 0 to exit.      │",
"│1: Navigation & contents│",
"│2: What's Ur?           │",
"│3: General goal         │",
"│4: Rules α to δ         │",
"│5: Rules ε to η         │",
"│6: PyUr's output        │",
"│7: PyUr's input         │",
"│8: Credits              │",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│2: What's Ur?           │",
"│                        │",
"│Ur is|was an ancient    │",
"│racing game (like Back- │",
"│gammon or Ludo,) found  │",
"│in Mesopotamia.         │",
"│                        │",
"│Its original rules are  │",
"│not known. This program │",
"│uses the rules proposed │",
"│by Irving Finkel of the │",
"│British Museum.         │",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│3: General goal         │",
"│                        │",
"│The aim of the game for │",
"│both players is to get  │",
"│all of their pieces in  │",
"│the end zone. Starting  │",
"│at their home, each pie-│",
"│ce must cross the middle│",
"│row of the board along- │",
"│side enemy pieces. Move-│",
"│ment is dictated by the │",
"│sum of four binary die. │",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│4: Rules α to δ         │",
"│                        │",
"│α: White goes first.    │",
"│β: You must move unless │",
"│ you absolutely can't.  │",
"│γ: Moves that land on an│",
"│opponent piece are valid│",
"│and will move said piece│",
"│to position 0, if it is │",
"│not on a rosetta.       │",
"│δ: Pieces are available │",
"│to leave home whenever. │",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│5: Rules ε to η         │",
"│                        │",
"│ε: If a move lands on a │",
"│rosetta, the player mov-│",
"│es again, no re-rolling.│",
"│ζ: For a piece to exit, │",
"│the exact number to land│",
"│on the exit tile must be│",
"│rolled. Otherwise, they │",
"│can't move on that turn.│",
"│η: You may only move on-│",
"│wards by what is rolled.│",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│6: PyUr's output        │",
"│                        │",
"│The cyan and magenta    │",
"│tiles are exit and      │",
"│entrance tiles, respect-│",
"│ively. They show their  │",
"│\"population.\" All pieces│",
"│that have left the game │",
"│will be displayed on the│",
"│line by the exit. Pieces│",
"│on the field are shown  │",
"│as a white|black number.│",
"│                        │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│7: PyUr's input         │",
"│                        │",
"│An M means the player   │",
"│must press one of the   │",
"│allowed — as will be    │",
"│told — numbers from 1 to│",
"│6 to move a piece. A W  │",
"│means that the program  │",
"│waits for the player to │",
"│press 0 to continue, or,│",
"│when valid, 7 to 9.     │",
"│'T' means you must type.│",
"│7=Save, 8=Load, 9=Help. │",
"└────────────────────────┘"],
[
"┌────────────────────────┐",
"│8: Credits              │",
"│                        │",
"│PyUr — copyleft 2017 —  │",
"│is by  Henry \"wolfboyft\"│",
"│Fleminger Thomson, and  │",
"│licensed under the GNU  │",
"│GPL 3. Thank you to the │",
"│ancient people who made │",
"│the original board, the │",
"│people who discovered it│",
"│and Irving Finkel for   │",
"│making such good rules. │",
"│                        │",
"└────────────────────────┘"]]

message = "" # Message backup
status = "" # Status backup

def manual():
	selection = 1
	while selection != 0:
		for i in range(len(help[selection - 1])): # Not - 1 because the upper limit of a range is exclusive.
			for j in range(len(help[selection - 1][i])):
				try:
					stdscr.addch(i + gamey, j + gamex, help[selection - 1][i][j]) # addstr outputs trash
				except curses.error:  # Bottom-right corner error.
					pass
		stdscr.refresh()
		char = stdscr.getch()
		if chr(char) in "012345678":
			selection = int(chr(char))
	
	for i in range(len(game)):
		for j in range(len(game[i])):
			try:
				stdscr.addch(i + gamey, j + gamex, game[i][j]) # addstr outputs trash
			except curses.error:  # Bottom-right corner error.
				pass
	
	# The stdscr.refresh() within refresh_board which is called after manual() will deal with re-printing the game.
	
	output(0, None)
	output(1, None)
	# Except for the outputted messages.

# Often functions that take the moves as an argument call it choices.
def tell_choices(choices):
	yes = []
	for i in range(len(choices)):
		if choices[i] is not False:
			yes.append(str(i + 1)) # Plus one for the user output (board[turn][0] is piece 1!)
	
	if len(yes) <= 4:
		return '|'.join(yes)
	elif len(yes) == 5:
		for i in range(1, 7):
			if str(i) not in yes:
				return "all - " + str(i) + "."
	elif len(yes) == 6:
		return "any one."

def get_player(whomst):
	if whomst:
		return white
	return black

def output(which, string = None, special = False): # Print a string to the status (which == 0) or message (which == 1) box, or backup/restore any strings in their respective variables.
	global status
	global message
	
	if string is None: # Restore
		if which == 0:
			output(0, status)
		else: # which == 1
			output(1, message)
	elif special: # Backup
		if which == 0:
			status = string
		else:
			message = string
	
	if string is not None:
		# We still print it, of course. Nothing in this program needs to backup but not print.
		if len(string) > 24:
			string = string[:24]
		
		stdscr.move(gamey + 11 + which * 2, gamex + 1)
		stdscr.clrtoeol()
		stdscr.addstr(string)
		stdscr.addch(gamey + 11 + which * 2, gamex + 25, '│')
		stdscr.refresh()

def get_moves(player, opponent, speed):
	if speed == 0: # No move that can be made will have any effect, so why allow any?
		return [False] * 6
	
	# moves[i] == False means that you can't move piece i
	# moves[i] == True means that you can move piece i
	# type(moves[i]) == int means that you can move piece i, and reset piece move[i] on the opponent's side
	moves = []
	
	# Get a piece.
	for i in range(6):
		destination = player[i] + speed
		# Check the validity of the spot it would go to.
		if destination <= 14 and destination not in player: # Check for being within board bounds and for collision with friendly pieces.
			# The non-shared positions have the same numbers for both players. "Home" positions (0 to 4 and 13 to 15) must be excluded from the enemy collision check.
			if 5 <= destination <= 12 and destination in opponent: # Is the destination going to hit an enemy that is not on a rosetta?
				if destination not in rosettas:
					moves.append(destination)
				else:
					moves.append(False)
			else:
				moves.append(True) # Add legal move to list.
		elif destination == 15: # Pieces can stand with their allies in the entrance and exit tiles.
			moves.append(True)
		else:
			moves.append(False) # Add illegal move to list.
	return moves

def is_move(x): # filter(is_move, moves)
	return x is not False

def main(stdscr): # Contains the definitions for choose_move, wait, save and load so that they can access main's variables.
	global turn # When loading, only the board can be modified. Something to with the fact that it's within a list...? But turn, roll and finished can't be changed from inside the wait function like that, so... yeah.
	global finished # In fact, for some reason, they have to be defined as global under wait's definition as well as here!! Beats me.
	global roll
	
	saveable = False
	loadable = False
	
	def choose_move(choices):
		stdscr.addch(12 + gamey, 23 + gamex, 'M')
		acceptables = []
		for i in range(6): # Extract the actual movable pieces' IDs from the move data.
			if choices[i]:
				acceptables.append(i + 1) # Raise the acceptable pieces' IDs up by one to meet... [WORMHOLE]
			# else we're looking at a datum for an invalid move, the existence of which is explained by the first few comments in get_moves.
		
		while True:
			try:
				choice = int(chr(stdscr.getch())) # [WORMHOLE] ...the user's input.
				if choice == 7 and saveable: # if chr(char) == '7':
					save()
				# It's never loadable at this time.
				elif choice == 9:
					manual()
					stdscr.addch(12 + gamey, 23 + gamex, 'M')
					refresh_board()
			except ValueError: # Numbers only!! (int('x') would raise ValueError.)
				choice = None # That sounds about right.
			
			if choice in acceptables:
				return choice - 1 # And lower it back down for the program!
	
	def refresh_board():
		def has_stone(position, whomst):
			if position in board[whomst]:
				return str(board[whomst].index(position) + 1)
			return ' '
		
		for i in range(2): # i is a player
			for j in range(6): # j is a piece
				if board[i][j] == 15:
					stdscr.addch(gamey + 9 - (8 * (i)), gamex + 3 + j * 2, str(j + 1))
				else:
					stdscr.addch(gamey + 9 - (8 * (i)), gamex + 3 + j * 2, str('─'))
			
			# Leaving home
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 14, str(board[i].count(0)), curses.color_pair(6 - i)) # entrance
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 11, has_stone(1, i), curses.color_pair(2 - i + (2 * (1 in rosettas))))
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 8, has_stone(2, i), curses.color_pair(2 - i + (2 * (2 in rosettas))))
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 5, has_stone(3, i), curses.color_pair(2 - i + (2 * (3 in rosettas))))
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 2, has_stone(4, i), curses.color_pair(2 - i + (2 * (4 in rosettas))))
			
			# Going home
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 23, has_stone(13, i), curses.color_pair(2 - i + (2 * (13 in rosettas))))
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 20, has_stone(14, i), curses.color_pair(2 - i + (2 * (14 in rosettas))))
			stdscr.addch(gamey + 8 - (6 * (i)), gamex + 17, str(board[i].count(15)), curses.color_pair(8 - i)) # exit
		
		for i in range(5, 13): # i is a position
			if i in board[True]:
				stdscr.addch(gamey + 5, gamex + (i - 4) * 3 - 1, str(board[True].index(i) + 1), curses.color_pair(1 + (2 * (i in rosettas))))
			elif i in board[False]:
				stdscr.addch(gamey + 5, gamex + (i - 4) * 3 - 1, str(board[False].index(i) + 1), curses.color_pair(2 + (2 * (i in rosettas))))
			else:
				stdscr.addch(gamey + 5, gamex + (i - 4) * 3 - 1, ' ', curses.color_pair(1 + (2 * (i in rosettas))))
		
		stdscr.refresh()
		curses.curs_set(False) # Y'ought not to resize the terminal.
	
	def save():
		code = 0
		
		for i in range(2):
			for j in range(6):
				code |= board[i][j] << ((6 * i + j ) * 4) # OR seems nicer than addition, I guess... we're going to be using AND to get 'em out, anyway!
		
		code |= turn << 52
		code |= finished << 51
		code |= roll << 48 # roll is 3-bit
		output(0, "Code: " + str(code))
		wait(False) # Don't allow this to be called again from wait()
		output(0, special = False) # Restore
	
	def wait(allow_specials = True):
		global turn
		global finished
		global roll
		
		stdscr.addch(12 + gamey, 23 + gamex, 'W')
		while True:
			char = stdscr.getch()
			if char == ord('7') and saveable and allow_specials: # if chr(char) == '7':
				save()
			elif char == ord('8') and loadable and allow_specials:
				output(1, "Enter code to load.") # A 53-bit code. Crikey!
				stdscr.addch(12 + gamey, 23 + gamex, 'T')
				code = "" # There's no copy and paste, but you could always put the code in here before running the program.
				char = None
				while True:
					output(0, code)
					char = stdscr.getch()
					if char == 8 and len(code): # Backspace. Make sure it doesn't work if the length of the code is zero!!
						code = code[:-1] # Delete the last character.
					elif len(code) < 16 and chr(char) in "0123456789": # It's never going to be bigger than 16, but we'll use >= anyway. (len(str(2 ** 53 - 1)) == 16)
						code += chr(char)
						output(0, code)
					elif char == ord('\n'):
						break # Enter to finish.
					# This beep thing was super-unreliable... might've been due to the nature of getch and some kind of queue or something... but yeah, I've removed it.
					# elif char is not None: # The first time we enter this loop, char is None.
					# 	curses.beep()
				
				try:
					code = int(code) # The potential exception-raiser.
					
					for i in range(2):
						for j in range(6):
							board[i][j] = (code & (15 << ((6 * i + j ) * 4))) >> ((6 * i + j) * 4)
					
					turn = bool((code & (1 << 52)) >> 52)
					finished = bool((code & (1 << 51)) >> 51)
					roll = (code & (7 << 48)) >> 48
					
					refresh_board()
					output(1, "Code loaded!")
				except ValueError: # If you enter nothing.
					output(1, "Invalid code.")
				
				wait(False)
				output(0, None)
				output(1, None)
			elif char == ord('9') and allow_specials:
				manual()
				stdscr.addch(12 + gamey, 23 + gamex, 'W')
				refresh_board()
			elif char == ord('0'): # if chr(char) == '0':
				return True # Ready to move on.
	
	# False = Black
	# True = White
	# None = Game over
	turn = True
		
	# 0 = Black
	# 1 = White
	board = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
	
	moves = []
	move = 0
	roll = 0
	finished = False
	
	# stdscr is passed in already, no initialisiation required.
	
	curses.curs_set(False) # This works 'til you resize the window.
	
	# Yeah, I know... I should have made black come first... (False == 0 and 0 < 1)
	curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_YELLOW) # White on a game tile
	curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW) # Black on a game tile
	
	curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_RED) # White on a rosetta
	curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_RED) # Black on a rosetta
	
	curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_MAGENTA) # Black on an entry tile
	curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_MAGENTA) # Black on an entry tile
	
	curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_CYAN) # White on an exit tile
	curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN) # Black on an exit tile
	
	for i in range(len(game)):
		for j in range(len(game[i])):
			try:
				stdscr.addch(i + gamey, j + gamex, game[i][j]) # addstr outputs trash
			except curses.error:  # Bottom-right corner error.
				pass
	
	output(0, "PyUr", True)
	output(1, "Press 9 for help", True)
	loadable = True
	wait()
	
	loadable = False 
	saveable = True
	refresh_board()
	
	while not finished:
		roll = randint(0, 4)
		output(0, get_player(turn) + " rolled a " + str(roll) + ".", True)
		
		while True:
			saveable = False
			moves = get_moves(board[turn], board[not turn], roll)
			
			if all(moves[i] is False for i in range(len(moves))): # No moves
				refresh_board()
				output(1, get_player(turn) + "'s go is over.", True)
				saveable = True
				break
			elif len(list(filter(is_move, moves))) == 1: # One move
				output(1, get_player(turn) + " has to move " + tell_choices(moves) + ".", True)
				wait()
				for i in range(6):
					if moves[i] is not False:
						move = i
						break
				board[turn][move] += roll
			else: # Multiple moves
				output(1, get_player(turn) + " may use " + tell_choices(moves), True) # Backup because choose_move can engage the manual.
				move = choose_move(moves)
				board[turn][move] += roll
			
			if type(moves[move]) is int:
				board[not turn][board[not turn].index(moves[move])] = 0
			
			if board[turn][move] in rosettas:
				refresh_board()
				output(1, get_player(turn) + " must move again.", True)
				wait()
			else:
				roll = 0
		
		if all(piece == 15 for piece in board[turn]): # [Line marked as "win."]
			saveable = True
			finished = True
		else:
			turn = not turn # Swap turns.
			wait()
	
	# If you loaded a won game, then here you are.
	# If you finished a game from "win," then this'll work too.
	# refresh_board() # No need to do this if you didn't load to this point, but the loading routine (marked as "load") does provided you entered a valid code, so it is done there.
	output(0, get_player(turn) + " wins!", True)
	output(1, "Thanks, both of you.", True)
	wait()
	
	curses.curs_set(True)

stdscr = curses.initscr()
curses.wrapper(main)