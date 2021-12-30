#date: 2021-12-30T17:18:28Z
#url: https://api.github.com/gists/e53c40c26a90f3dc9b76660ff6e722bb
#owner: https://api.github.com/users/JB-Tellez

class Game:

    WAITING_FOR_PLAYER_ONE = 'WAITING_FOR_PLAYER_ONE'
    WAITING_FOR_PLAYER_TWO = 'WAITING_FOR_PLAYER_TWO'
    COLLECTING_INFO = 'COLLECTING_INFO'
    PLAYER_ONE_TURN = 'PLAYER_ONE_TURN'
    PLAYER_TWO_TURN = 'PLAYER_TWO_TURN'
    GAME_OVER = 'GAME_OVER'

    def __init__(self):
        self.state = self.WAITING_FOR_PLAYER_ONE
        self.player_one = None
        self.player_two = None
        self.current_player = None

    def process_connection(self, id):
        if self.state == self.WAITING_FOR_PLAYER_ONE:
            self.player_one = Player(id)
            self.state = self.WAITING_FOR_PLAYER_TWO
            return "Waiting for player 2"
        elif self.state == self.WAITING_FOR_PLAYER_TWO:
            self.player_two = Player(id)
            self.state = self.COLLECTING_INFO
            return 'Enter favorite number'
            
        
        return "This game full..."



    def process_message(self, msg, id):

        if msg == "quit":
            self.state = self.GAME_OVER
            return 'adios'
        elif self.state == self.COLLECTING_INFO:

            player = self.player_one if self.player_one.id == id else self.player_two

            player.fave = int(msg)

            if self.player_one.fave is not None and self.player_two.fave is not None:
                self.state = self.PLAYER_ONE_TURN

                self.current_player = self.player_one

                return 'player one turn'
            else:
                return "waiting on other player..."
            

        elif self.state == self.PLAYER_ONE_TURN:

            if id == self.current_player.id:
                result = self.process_guess(int(msg))
                if self.state != self.GAME_OVER:
                    self.state = self.PLAYER_TWO_TURN
                self.current_player = self.player_two
                return result
            else:
                return "Not your turn!"

        elif self.state == self.PLAYER_TWO_TURN:

            if id == self.current_player.id:
                result = self.process_guess(int(msg))
                if self.state != self.GAME_OVER:
                    self.state = self.PLAYER_ONE_TURN
                self.current_player = self.player_one
                return result
            else:
                return "Not your turn!"



        return self.state


    def process_guess(self, num):
        other_player = self.player_one if self.current_player == self.player_two else self.player_two
      
        if num == other_player.fave:
            self.state = self.GAME_OVER
            return f"Victory for {self.current_player}"
        elif num < other_player.fave:
            return "too low"
        else:
            return "too high"
        
        

class Player:
    def __init__(self, id):
        self.id = id
        self.fave = None

    def __str__(self):
        return f"{self.id}:{self.fave}"