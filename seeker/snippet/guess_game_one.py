#date: 2021-12-22T17:11:13Z
#url: https://api.github.com/gists/33d2b6319b046127df86c57b9f921d37
#owner: https://api.github.com/users/PythonRulz

'''
	Generate a random number between 1 and 9 (including 1 and 9). Ask the user to guess the number, 
	then tell them whether they guessed too low, too high, or exactly right.

Extras:

Keep the game going until the user types “exit”
Keep track of how many guesses the user has taken, and when the game ends, print this out.
'''


import random

def get_number():
    return random.randint(1,9)
    

def main():
    secret_number = get_number()
    print(secret_number)
    guesses = 1
    play = True
    while play:
        user_guess = int(input("Guess the number: between 1 and 9: "))
        if user_guess < 1 or user_guess > 9:
            print("1 through 9 only please")
        elif user_guess > secret_number:
            print("That guess is to high, try again")
            print()
            guesses += 1
        elif user_guess < secret_number:
            print("That guess is to low, try again")
            print()
            guesses += 1
        elif user_guess == secret_number:
            print(f"That's it, the secret number was {secret_number}")
            if guesses > 1:
                print(f"It took you {guesses} guesses")
            else:
                print(f"It only took you {guesses} guess")
            play = False
            print()
			
    answer = input("Type 'exit' to quit or anything else to play again: ").lower()
    while True:
        if answer == '':
            main()
        elif answer == 'exit':
            break
        else:
            answer = input("Type 'exit' to quit or anything else to play again: ")
        
        
main()