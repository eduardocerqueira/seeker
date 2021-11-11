#date: 2021-11-11T17:01:42Z
#url: https://api.github.com/gists/d8fc22687cd57328fd0b916ecc7884f3
#owner: https://api.github.com/users/KarloSegota

import datetime
import json
import random

player = input("Please enter your name: ")

secret = random.randint(1, 30)
attempts = 0

with open("score_list.json", "r") as score_file:
    score_list = json.loads(score_file.read())

for score_dict in score_list:
    score_text = f"Player {score_dict['player_name']} had {score_dict['attempts']} attempts on {score_dict['date']}." \
                 f"The secret number was {score_dict['secret_number']}." \
                 f"And wrong guesses were {score_dict['wrong_guesses']}"

wrong_guesses = []
while True:
    guess = int(input("Guess the secret number between 1 and 30: "))
    attempts += 1

    if guess == secret:
        score_list.append({"player_name": player, "attempts": attempts, "date": str(datetime.datetime.now()),
                           "secret_number": secret, "wrong_guesses": wrong_guesses})
        with open("score_list.json", "w") as score_file:
            score_file.write(json.dumps(score_list))

        print("your guess is correct")
        print("attempts needed: " + str(attempts))
        break

    elif guess > secret:
        print("Try smaller")

    elif guess < secret:
        print("Try larger")

    wrong_guesses.append(guess)
