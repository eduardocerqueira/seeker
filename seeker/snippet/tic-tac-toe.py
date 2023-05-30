#date: 2023-05-30T16:48:53Z
#url: https://api.github.com/gists/30dfcbc24fcb02419cc7f0a75f2a2259
#owner: https://api.github.com/users/MipoX

class Player:
    """
    Создается объект игрока, которому можно присваивать имя.
    Игрок имеет статус, который определяет его как победителя, по умолчанию False
    Так же игроку присваивается подпись.
    Игрок может на игровом поде(board) оставлять свою подпись,
    тем самым он совершает ход и вносит изменения в игровом поле
    """

    def __init__(self, name, board):
        self.name = name
        self.status = False
        self.signature = None
        self.board = board

    def __repr__(self):
        return f' {self.name}; Подпись: {self.signature}'

    def progress(self):
        try:
            player_signature = int(input('Выберите клетку: '))
            if player_signature < 1 or player_signature > 9:
                raise NameError()
            else:
                if player_signature in self.board.marked_moves:
                    print(f'Эта клетка уже занята.')
                    self.progress()
                else:
                    for elem in self.board.playing_field:
                        if player_signature in elem:
                            index = elem.index(player_signature)
                            elem[index] = self.signature
                            self.board.marked_moves.append(player_signature)
                            return


        except ValueError:
            print(f'Ошибка! Необходимо ввести целое число.')
            self.progress()
        except NameError:
            print(f'Ошибка! Необходимо выбирать номер клетки из указанных на поле!.')
            self.progress()


class Board:
    """
    Класс используется как игровое поле для участников игры.
    """

    def __init__(self):
        self.playing_field = [list(range(i, i + 3)) for i in range(1, 9, 3)]
        self.marked_moves = list()

    def __repr__(self):
        return f'Использованные ходы: {self.marked_moves}'

    def print_matrix(self):
        line_print = f"\n{'_' * 19:^0}\n"
        matrix = ''.join(line_print)
        for row in self.playing_field:
            for element in row:
                matrix += ''.join(f"|{element:^5}")
            matrix += ''.join('|' + line_print)
        return matrix


class CheckBoard:
    """
    Класс для проверки ходов игроков. Запускается после 5 хода.
    Проверка проходит по горизонтали, вертикали и по двум диагоналям.
    Возвращает True or False
    """

    def __init__(self, board: Board, player_1: Player, player_2: Player):
        self.check_board = board.playing_field
        self.player_1 = player_1
        self.player_2 = player_2

    def check(self):
        if any([self.line_check(), self.elem_of_index_check(), self.diagonal_to_right_check(),
                self.diagonal_to_left_check()]):
            return False
        return True

    def line_check(self):
        # Проверяем строку
        for elem in self.check_board:
            if all(True if i_elem == "X" else False for i_elem in elem):
                self.player_1.status = True
                return True
            elif all(True if i_elem == "O" else False for i_elem in elem):
                self.player_2.status = True
                return True

        return False

    def elem_of_index_check(self):
        # Проверяем построчно отдельный элемент, по индексу(проверяем совпадения 3 по вертикали)
        for index in range(3):
            count_player_1 = 0
            count_player_2 = 0
            for line in self.check_board:

                if line[index] == "X":
                    count_player_1 += 1
                    if count_player_1 == 3:
                        self.player_1.status = True
                        return True

                elif line[index] == "O":
                    count_player_2 += 1
                    if count_player_2 == 3:
                        self.player_2.status = True
                        return True

        return False

    def diagonal_to_right_check(self):
        # Проверка по диагонали слева
        digit_line = 0
        count_player_1 = 0
        count_player_2 = 0
        for line in self.check_board:
            if (digit_line == 2 and count_player_2 == 0) and (digit_line == 2 and count_player_1 == 0):
                return False
            else:
                if line[digit_line] == "X":
                    count_player_1 += 1
                    if count_player_1 == 3:
                        self.player_1.status = True
                        return True

                elif line[digit_line] == "O":
                    count_player_2 += 1
                    if count_player_2 == 3:
                        self.player_2.status = True
                        return True
            digit_line += 1

        return False

    def diagonal_to_left_check(self):
        # Проверка по диагонали справа
        digit_line_rev = -1
        count_player_1 = 0
        count_player_2 = 0
        for line in self.check_board:
            if (digit_line_rev == -2 and count_player_2 == 0) and (digit_line_rev == -2 and count_player_1 == 0):
                return False
            else:
                if line[digit_line_rev] == "X":
                    count_player_1 += 1
                    if count_player_1 == 3:
                        self.player_1.status = True
                        return True

                elif line[digit_line_rev] == "O":
                    count_player_2 += 1
                    if count_player_2 == 3:
                        self.player_2.status = True
                        return True
                digit_line_rev += -1
        return False


class Game:
    """
    В данном классе проходит сам игровой процесс, создаются объекты игроков,
    игрового поля. Игрокам присваиваются подписи.
    Проводится подсчет ходов, проверяется статус игры(с 5 хода)

    """

    def __init__(self):
        self.board = Board()
        self.player_1 = Player('Kot_1', self.board)
        self.player_2 = Player('Kot_2', self.board)
        self.player_1.signature = 'X'
        self.player_2.signature = 'O'
        self.result = CheckBoard(self.board, self.player_1, self.player_2)
        self.vs = [self.player_1, self.player_2]
        self.count_move = 0

    def __repr__(self):
        return f'Игроки: {self.player_1.name, self.player_2.name} Ход № {self.count_move}'

    def start_game(self):
        while self.count_move != 9:
            for player in self.vs:
                if self.count_move == 9:
                    print(self.board.print_matrix())
                    print(self.print_result())
                    break

                if self.count_move >= 5:
                    if self.result.check():
                        self.next_move(player)
                    else:
                        self.count_move = 9
                else:
                    self.next_move(player)

    def next_move(self, player):
        print(self.board.print_matrix())
        print(f'Ходит: {player.name}')
        player.progress()
        self.count_move += 1

    def print_result(self):
        for player in self.vs:
            if player.status:
                return f'Победитель: {player.name}'
        return f'Ничья!'


a = Game()
a.start_game()
