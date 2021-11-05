#date: 2021-11-05T17:13:01Z
#url: https://api.github.com/gists/3a32d8e665e266580d05428dba56cf94
#owner: https://api.github.com/users/JakubDotPy

name = 'Testplayer'
character = 'x'


def ask_for_position():
    while True:
        try:
            pos = int(input('Enter the index where you want to play: '))
        except ValueError:
            print('Must be a number!')
        else:
            return pos


def choose_position(self, board):
    return self.ask_for_position()
