#date: 2021-11-05T17:13:01Z
#url: https://api.github.com/gists/3a32d8e665e266580d05428dba56cf94
#owner: https://api.github.com/users/JakubDotPy

def main():
    kuba = Human(name='Kuba', character='x')
    
    # this does not work
    # TypeError: Can't instantiate abstract class Player with abstract methods choose_position 
    imported_player = Player.from_file('some_player') 

    players = [kuba, imported_player]
    board = Board(size=20, empty_character='-')

    tic_tac_toe = Game(players, board)

    tic_tac_toe.print_intro()
    tic_tac_toe.play()
    tic_tac_toe.print_outro()


if __name__ == '__main__':
    exit(main())