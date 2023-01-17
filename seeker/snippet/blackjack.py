#date: 2023-01-17T16:58:27Z
#url: https://api.github.com/gists/392dc4658cbd6cd814ef4d920ef080c1
#owner: https://api.github.com/users/GamerztheCoder

import random
import time

def card_calculator(hand):
    total = 0
    for card in hand:
         card_info = card.split(' ')
         if card_info[0] == 'Ace':
            print('You have an Ace in your hand.')
            ace_total = input('How many points will the Ace give you? (1 or 11)  ')
            if ace_total == '11':
                total += 11
            elif ace_total == '1':
                total += 1
            else:
                print('Picking number for you...')
                ace_total = random.randint(1,2)
                if ace_total == 1:
                    total += 1
                else:
                    total += 11
         elif card_info[0] == 'King':
            total += 10
         elif card_info[0] == 'Queen':
            total += 10
         elif card_info[0] == 'Jack':
            total += 10
         else:
            total += int(card_info[0])
    return total
def dealer_card_calculator(hand_deck):
    total = 0
    for card in hand_deck:
        card_info = card.split(' ')
        if card_info[0] == 'Ace':
            if total <= 10:
                total += 11
            else:
                total += 1
        elif card_info[0] == 'King':
            total += 10
        elif card_info[0] == 'Queen':
            total += 10
        elif card_info[0] == 'Jack':
            total += 10
        else:
            total += int(card_info[0])
    return total
def show_cards(deck):
    string = ''
    for item in deck:
        string += item
        string += ', '
    print(string)
def deal_cards(list):
    for item in range(2):
        card = random.choice(DECK)
        while card in USED_CARD:
            card = random.choice(DECK)
        else:
            USED_CARD.append(card)
            list.append(card)
def add_card(deck_hand):
    card = random.choice(DECK)
    while card in USED_CARD:
        card = random.choice(DECK)
    else:
        deck_hand.append(card)

DECK = []
USED_CARD = []

titles = ['Diamonds', 'Hearts', 'Clubs', 'Spades']
tYPe = ['King', 'Queen', 'Jack', '10', '9', '8', '7', '6', '5', '4', '3', '2', 'Ace']

for royal in titles:
    for title in tYPe:
        card = title + ' of ' + royal
        DECK.append(card)

current_hand = []
dealers_hand = []
deal_cards(current_hand)
deal_cards(dealers_hand)
current_hand_total = card_calculator(current_hand)
print('Player\'s Turn:')
print()
print('Opponents\'s Second Card: ' + dealers_hand[1])
print()
show_cards(current_hand)
print('Your Total: ' + str(current_hand_total))
print()
if current_hand_total < 21:
    hit_or_stay = 'hit'
else:
    hit_or_stay = 'stay'
while hit_or_stay != 'stay' and current_hand_total <= 21:
    hit_or_stay = input('Would you like to hit or stay? ')
    if hit_or_stay == 'hit':
        add_card(current_hand)
        current_hand_total = card_calculator(current_hand)
        show_cards(current_hand)
        print('Your Total: ' + str(current_hand_total))
        print()
        if current_hand_total == 21:
            print('You have hit 21!')
            break
        elif current_hand_total > 21:
            print('You BUSTED!')
            break
        else:
            pass
    else:
        print('You chose to stay.')
        break
print('Opponent\'s Turn: ')
time.sleep(1)
print()
show_cards(dealers_hand)
dealers_total = dealer_card_calculator(dealers_hand)
print('Opponent\'s Total: ' + str(dealers_total))
print()
time.sleep(2)
while dealers_total <= 16:
    print('Opponent chose to hit.')
    add_card(dealers_hand)
    dealers_total = dealer_card_calculator(dealers_hand)
    show_cards(dealers_hand)
    print('Opponent\'s Total: ' + str(dealers_total))
    print()
    time.sleep(3)
else:
    print('Opponent chose to stay.')
    time.sleep(2)
print()
print('Your Total: ' + str(current_hand_total))
print('Opponent\'s Total: ' + str(dealers_total))
print()
if dealers_total > 21 and current_hand_total > 21:
    print('Both you and your opponent busted.')
elif dealers_total >= 22:
    print('Opponent Busted. You Won.')
elif current_hand_total >= 22:
    print('You Busted. Opponent Won.')
elif dealers_total == current_hand_total:
    print('You have tied.')
elif dealers_total > current_hand_total:
    print('Opponent Won.')
elif dealers_total < current_hand_total:
    print('You Won.')


