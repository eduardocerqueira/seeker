#date: 2022-07-18T17:15:38Z
#url: https://api.github.com/gists/62497e1e7a6c744ae4a0e5662830715d
#owner: https://api.github.com/users/Riddle1001

import queue
import time
from ctypes import create_unicode_buffer, windll

import keyboard
from rlbot.agents.base_script import BaseScript
from rlbot.utils.game_state_util import GameInfoState, GameState


def is_rocket_league_focused() -> bool:
    # https://stackoverflow.com/questions/10266281/obtain-active-window-using-python
    hWnd = windll.user32.GetForegroundWindow()
    length = windll.user32.GetWindowTextLengthW(hWnd)
    buf = create_unicode_buffer(length + 1)
    windll.user32.GetWindowTextW(hWnd, buf, length + 1)
    title = buf.value or ""
    return title.startswith("Rocket League")


def ignore_when_rl_not_focused_wrapper(func):
    def wrapper():
        if is_rocket_league_focused():
            func()

    return wrapper


class MyScript(BaseScript): # thanks Darxeal for providing the information to create this script 
    def __init__(self):
        super().__init__("time machine")
        
        print("time machine init")
        
        self.states = []
        self.state_i = 0
        self.gameplay_paused = False
        self.state_rate = 1 # number of seconds to wait before adding a state
        self.unpause_speed = 1
        self.last_state_added = time.time()

        self.ois_kickoff_pause = False
      
        keyboard.add_hotkey('down', self.pause, args=(False, self.unpause_speed))
        keyboard.add_hotkey("right", self.change_state, args=(1,))
        keyboard.add_hotkey("left", self.change_state, args=(-1,))

    def remove_future_ticks(self): # remove all states after the current one
        self.states = self.states[:self.state_i + 1]

    def pause(self, b: bool, delay: int = 0):
        if delay:
             time.sleep(delay)
             self.remove_future_ticks()
        self.set_game_state(GameState(game_info=GameInfoState(paused=b)))
        self.gameplay_paused = b
    
    def toggle_pause(self):           
        self.pause(not self.gameplay_paused)

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)
        

    def change_state(self, index: int):
        if not self.gameplay_paused:
            self.pause(True)
        # Clamp between 0 and len(self.states) - 1
        index = self.clamp(index + self.state_i, 0, len(self.states) - 1)
        self.state_i = index
        self.toggle_pause()
        self.wait_game_tick_packet()
        self.set_game_state(self.states[self.state_i])
        self.toggle_pause()

    def reset(self):
        self.states = []
        self.state_i = 0
        self.last_state_added = time.time()

    def run(self):
        while True:
            packet = self.wait_game_tick_packet()
            if self.ois_kickoff_pause != packet.game_info.is_kickoff_pause:
                self.reset()
                self.ois_kickoff_pause = packet.game_info.is_kickoff_pause

            current_game_state = GameState.create_from_gametickpacket(packet)
            active_game = packet.game_info.is_round_active and not packet.game_info.is_match_ended and not self.gameplay_paused
            if active_game and time.time() - self.last_state_added > self.state_rate:
                self.states.append(current_game_state)
                self.last_state_added = time.time()
                self.state_i = len(self.states) - 1
            

if __name__ == "__main__":
    script = MyScript()
    script.run()