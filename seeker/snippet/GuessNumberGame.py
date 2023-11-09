#date: 2023-11-09T16:37:11Z
#url: https://api.github.com/gists/f2b6c3ffee0fc6d5047d8023be3acaa1
#owner: https://api.github.com/users/PhiLongVu

"""
This program generates a random number and asks the player to guess
"""
import random

MIN_GUESS = 1
MAX_GUESS = 100

def generate_random_number():
    """
    generate a random number from min to max
    :return: an integer from the given range
    """
    #generates random number from min to max
    number = random.randint(MIN_GUESS, MAX_GUESS)
    return number

def get_player_guess_number():
    """
    asks the player to input a number, loops until the input is an integer
    :return:
    """
    number = ""
    is_continue_to_ask = True
    #loops until the number is a valid integer between min and max
    while(is_continue_to_ask):
        number = input("Please enter a number from " + str(MIN_GUESS) + " to " + str(MAX_GUESS) + ": ")
        try:
            number = int(number)
            if number <= MAX_GUESS and number >= MIN_GUESS:
                is_continue_to_ask = False
            else:
                print("The number must be between", MIN_GUESS, "to", MAX_GUESS)
        except:
            print("Please enter an valid whole number from", MIN_GUESS, "to", MAX_GUESS)
    return number
  
def give_feedback(guess_number, real_number):
    """
    gives feedback according to how higher or lower the guessing number is than the real number
    :param guess_number: player's number
    :param real_number: generated number
    """
    #compares the difference of both numbers and prints feedback
    difference = real_number - guess_number
    if difference > 0:
        print("Your number is too low")
    if difference < 0:
        print("Your number is too high")
    if difference == 0:
        print("Correct! You win")

def play_game():
    """
    simulates the game
    """
    #generates the number
    number_to_guess = generate_random_number()
    #for testing purpose, the real number is printed
    print("The real number generated is:", number_to_guess)
    is_player_continue = True
    #loops and gives feedback until the guesisng number equals to the real number
    while(is_player_continue):
        number_input = get_player_guess_number()
        if number_input == number_to_guess:
            is_player_continue = False
        give_feedback(number_input, number_to_guess)

#starts the game
play_game()



