#date: 2022-06-08T17:12:56Z
#url: https://api.github.com/gists/335a568489274173c8e081bb6822d978
#owner: https://api.github.com/users/tjhartline

#A dictionary for the simplified dragon text game
#The dictionary links a room to other rooms.
rooms = {
        'Great Hall': {'South': 'Bedroom'},
        'Bedroom': {'North': 'Great Hall', 'East': 'Cellar'},
        'Cellar': {'West': 'Bedroom'}
    }
print('\n' * 10)
print('Your starting and current location is the Great Hall.\n')
# noinspection PyRedeclaration
current_room = 'Great Hall'
# noinspection PyRedeclaration
validate = {'up': 'north', 'down': 'south', 'right': 'east', 'left': 'west', 'exits': 'exit'}
while current_room in 'Great Hall':
    next_move = str(input('You can only travel south from the Great Hall. \nEnter south to continue or exit:'))
    while next_move not in validate['down'] and next_move not in validate['exits']:
        print('\nInvalid entry.\n')
        next_move = str(input('\nEnter south or exit:'))
        continue
    if next_move in validate['down']:
        print('\nYou are now in the Bedroom.')
        current_room = 'Bedroom'
    else:
        next_move in validate['exits']
        exit('\nThanks for playing. Goodbye.')
    while current_room in 'Bedroom':
        next_move = str(input('\nEnter north or east to continue or exit to exit game:\n'))
        while next_move != validate['up'] and next_move != validate['right'] and next_move != validate['exits']:
            print('Invalid entry.')
            next_move = str(input('\nEnter north or east to continue or exit to exit game:\n'))
            continue
        if next_move == validate['up']:
            current_room = 'Great Hall'
            print('You are now back in the', current_room)
        elif next_move == validate['right']:
            current_room = 'Cellar'
            print('You are now in the', current_room)
        else:
            next_move = 'exit'
            exit('Thank you playing. Goodbye.')
    while current_room in 'Cellar':
        next_move = str(input('\nEnter west to continue or exit to exit game:\n'))
        while next_move != validate['left'] and next_move != validate['exits']:
            print('Invalid entry.')
            next_move = str(input('\nEnter west to continue or exit to exit game:\n'))
            continue
        if next_move == validate['left']:
            current_room = 'Bedroom'
            print('You are now in the', current_room)
        else:
            next_move = 'exit'
            exit('Thank you for playing. Goodbye')