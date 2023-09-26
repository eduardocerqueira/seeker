#date: 2023-09-26T16:54:27Z
#url: https://api.github.com/gists/49384ee0637a43af763df37ca2904bb6
#owner: https://api.github.com/users/oconnoob

import random  # imports at top, two empty lines after imports


# use constants for configurability
MIN_NUM = 0
MAX_NUM = 100

# break out into functions
def get_input():
    while True:
        try:
            query = input("what number am I thinking of? ")
            return int(query)
        # verify input with try/except
        except ValueError:
            print(f"Please enter a valid integer. You entered {query}")

def play_round(name):
    m_n = random.randrange(0, MAX_NUM-1)
    ticker = 0 
    while True:
        # take the ticker out of if/elif/else because it was common to all of them
        ticker += 1
        query = get_input()
        if query > MAX_NUM:
            print(f"the integer can't be greater than {MAX_NUM}") 
        elif query < MIN_NUM:
            print(f"the integer can't be lesser than {MIN_NUM}")
        elif query < m_n:
            print('not quite, the number is greater than your guess')
        elif query > m_n:
            print('not quite, the number is lesser than your guess')
        else:
            print(f"That's right {name}, the number is {m_n}")
            print(f"you tried {ticker} times")
            break

def ask_play_again():
    while True:
        play_again = input('would you like to play again? (y/n) ')         
        if play_again == 'y':
            return True
        elif play_again == 'n':
            return False
        # error handling for invalid option
        else:
            print('please enter a valid option')              

def play_game():
    name = input("what's your name? ")
    play_round(name)
    # ask_play_again() evaluated each time
    while ask_play_again():
        play_round(name)


# play the game only if this is executed as the main script (and not imported if you wanted to use these functions elsewhere)
if __name__ == '__main__':
    play_game()
