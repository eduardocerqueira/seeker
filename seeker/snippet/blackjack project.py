#date: 2022-12-16T16:47:44Z
#url: https://api.github.com/gists/deec7477e1383edcc82af2082e2b539e
#owner: https://api.github.com/users/mohitthakur9901

import random
from hangman import  blackjack
def deal_card(cards):
    cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
    card = random.choice(cards)
    return card
def game():
    print(blackjack)
    print("WELCOME TO BLAKJACK")
    user_cards = []
    computer_cards = []
    def calculate_score(cards):
        if sum(cards) == 21 and len(cards) == 2:
            return 0
        if 11 in cards and sum(cards) > 21:
            cards.remove(11)
            cards.append(1)
            return sum(cards)

    def compare(user, computer):
        if user == computer:
            return "drow match"
        elif computer == 0:
            return "computer won it's blackjack"
        elif user == 0:
            return "u=you win it's blackjack"
        elif user > 21:
            return "you lose"
        elif computer > 21:
            return "you win the match"
        elif user > computer:
            return "user win "
        else:
            return "computer win"
    for _ in range(2):
        user_cards.append(deal_card(0))
        computer_cards.append(deal_card(0))
        computer = sum(computer_cards)

    game_end = False

    while not game_end:
        user = sum(user_cards)
        print(f"your card is {user_cards} current score {user}")
        print(f"computer's first card is {computer_cards[0]}")
        if user == 0 and computer == 0 and user > 21:
            game_end = True
        else:
            user_choice = input("type 'yes'for another card and 'no' for pass \n")
            if user_choice == "yes":
                user_cards.append(deal_card(0))
            else:
                game_end= True
    while computer == 0 and computer < 17:
        computer_cards.append(deal_card(0))
        computer = calculate_score(computer_cards)

    print(f"your cards {user_cards} and score {user}")
    print(f"computer cards{computer_cards} and score {computer}")
    print(compare(user,computer))
while input("do you want to play blackjack type YES otherwise NO\n") == "yes":
    game()