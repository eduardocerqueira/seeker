#date: 2023-01-12T17:19:26Z
#url: https://api.github.com/gists/c8c6ce652a97ede3b141fb4eabd6e9c9
#owner: https://api.github.com/users/alanwnl

class ToyRobot:
    def __init__(self):
        self.allowed_actions = ["move", "speak", "dance"]
        self.allowed_properties = ["color", "size"]
    
    def receive_message(self, message):
        if message[0] in self.allowed_actions:
            getattr(self, message[0])(message[1])
        else:
            print("Invalid action")
    
    def move(self, direction):
        print("Moving", direction)
    
    def speak(self, phrase):
        print("Speaking", phrase)
    
    def dance(self):
        print("Dancing")
        
class User:
    def __init__(self):
        self.robot = ToyRobot()
        
    def send_message(self, action, argument=None):
        self.robot.receive_message((action, argument))
        
user = User()
user.send_message("move", "forward") # Output: Moving forward
user.send_message("speak", "hello") # Output: Speaking hello
user.send_message("dance") # Output: Dancing