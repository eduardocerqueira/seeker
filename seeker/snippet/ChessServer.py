#date: 2022-06-13T17:12:21Z
#url: https://api.github.com/gists/0e6d2414ca6bf06c6dc8a7c2df6ff0b1
#owner: https://api.github.com/users/Taonga07

from ChessEngine import Pawn, Rook, Bishop, Queen, King, Knight
from socket import socket, AF_INET, SOCK_STREAM
from os.path import join, dirname
from json import loads, load
import threading
import sys

class Server():
    def __init__(self, port=4442) -> None:
        self.host, self.port = "localhost", port
        self.server_active, self.games = True, []

    def __call__(self) -> None:
        self.start_sock()
        self.client_handler()

    def start_sock(self) -> None:
        self.sock_server = socket(AF_INET, SOCK_STREAM)
        self.sock_server.bind(("", self.port))
        self.sock_server.listen(True)

    def client_handler(self) -> None:
        players, connected_players = [], 0
        addr = f"tcp://127.0.0.1:{self.port}"
        print(f"\nWaiting for a connection on {addr}")
        while self.server_active:
            try:
                while True:
                    for i in range(2):
                        connected_players += 1
                        connection = self.sock_server.accept()
                        players.append((connection, connected_players))
                        if (i == 1) and (len(players) == 2): 
                            self.games.append(ChessGame(players))
                            self.games[-1].start()
                    players = []
            except KeyboardInterrupt:
                print("\nShutting down server")
                for game, _  in enumerate(self.games):
                    self.games[game].active = False
                self.server_active = False
        self.sock_server.close()
        sys.exit()

class ChessGame(threading.Thread):
    def __init__(self, clients) -> None:
        threading.Thread.__init__(self)
        self.players = [threading.Thread(target=self.player, args=clients[i]) for i in range(2)]
        self.board, self.active, self.player_turn = self.load_board(), True, 0

    def run(self):
        [player.start() for player in self.player]
        while self.active:
            self.players[self.player_turn].waiting = False
            self.players[1-self.player_turn].waiting = True #other player
        self.white_player.active = False
        self.black_player.active = False

    def load_board(self, board = eval(open(join(dirname(__file__), "Board.txt")).read().rstrip().replace("/n", ""))):
        pieces = load(open(join(join(dirname(__file__), 'ChessPieces.json'))))
        for row_number, row in enumerate(board):
            for column_number, square in enumerate(row):
                if square == "0": square = None
                if square is not None:
                    piece_name, piece_colour = list(pieces[square].values())
                    piece = eval(f"{piece_name}('{piece_colour}', {column_number}, {row_number})")
                    board[row_number][column_number] = piece
        return board

    def player(self, connection, ID):
        print(f" ... connection established from {connection[0]}")
        active, waiting, mssg_id = True, True, 0
        while active:
            if not waiting:
                incoming_msg = loads(connection[0].recv(1024).decode())
                # all checks if false value in list
                if all(isinstance(i, list) for i in incoming_msg["board"]):
                    if all(len(i) == 8 for i in incoming_msg["board"]):
                        if isinstance(incoming_msg["board"], list):
                            incoming_board = self.load_board(incoming_msg["board"])
        print(f"Bye from Client{ID}")
        self.conn.close()
        sys.exit()

class Player(threading.Thread):
    def __init__(self, connection, ID) -> None:
        threading.Thread.__init__(self)
        self.conn, self.addr = connection   
       

    def update_output_buffer(self, chess_game=None):
        output_buffer ={}
        output_buffer["msg_id"] = self.mssg_id+1
        if chess_game is None:
            self.output_buffer["board"] = chess_game
            self.output_buffer["error_msg"] = None
        else:
            self.output_buffer["error_msg"] = chess_game

    def run(self):
        print(f" ... connection established from {self.conn}")
        self.output_buffer = {"msg_id": self.mssg_id+1}
        while self.active:
            self.send(self.output_buffer)
            try: incoming_msg = loads(self.conn.recv(1024).decode())
            except: self.active = False
            if incoming_msg != None:
                if incoming_msg["msg_id"] > self.mssg_id:
                    self.msg_id = incoming_msg["msg_id"]
                    self.input_buffer = incoming_msg
            else: self.active = False
        print(f"Bye from Client{self.ID}")
        self.conn.close()
        sys.exit()

chess_server = Server()
chess_server()