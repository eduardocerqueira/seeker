#date: 2025-06-02T17:06:26Z
#url: https://api.github.com/gists/b2c0cbf4c3eda6ce71430d07b36ae8a1
#owner: https://api.github.com/users/Pythonista7

#!/usr/bin/env python3
import os
import sys
import time
import select
import tty
import termios

class Pong:
    def __init__(self):
        self.width = 60
        self.height = 20
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = 1
        self.ball_dy = 1
        self.paddle1_y = self.height // 2 - 2
        self.paddle2_y = self.height // 2 - 2
        self.paddle_size = 4
        self.score1 = 0
        self.score2 = 0
        
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def draw(self):
        self.clear_screen()
        
        # Draw top border
        print('=' * (self.width + 2))
        
        for y in range(self.height):
            line = '|'
            for x in range(self.width):
                if x == self.ball_x and y == self.ball_y:
                    line += 'O'  # Ball
                elif x == 0 and self.paddle1_y <= y < self.paddle1_y + self.paddle_size:
                    line += '#'  # Left paddle
                elif x == self.width - 1 and self.paddle2_y <= y < self.paddle2_y + self.paddle_size:
                    line += '#'  # Right paddle
                elif x == self.width // 2:
                    line += '|'  # Center line
                else:
                    line += ' '
            line += '|'
            print(line)
            
        # Draw bottom border
        print('=' * (self.width + 2))
        print(f'Player 1: {self.score1}  |  Player 2: {self.score2}')
        print('Controls: W/S for left paddle, I/K for right paddle, Q to quit')
        
    def update_ball(self):
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Bounce off top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - 1:
            self.ball_dy = -self.ball_dy
            
        # Check paddle collisions
        if (self.ball_x == 1 and 
            self.paddle1_y <= self.ball_y < self.paddle1_y + self.paddle_size):
            self.ball_dx = -self.ball_dx
            
        if (self.ball_x == self.width - 2 and 
            self.paddle2_y <= self.ball_y < self.paddle2_y + self.paddle_size):
            self.ball_dx = -self.ball_dx
            
        # Check for scoring
        if self.ball_x <= 0:
            self.score2 += 1
            self.reset_ball()
        elif self.ball_x >= self.width - 1:
            self.score1 += 1
            self.reset_ball()
            
    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = -self.ball_dx  # Reverse direction
        
    def get_key(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None
        
    def move_paddles(self, key):
        if key == 'w' and self.paddle1_y > 0:
            self.paddle1_y -= 1
        elif key == 's' and self.paddle1_y < self.height - self.paddle_size:
            self.paddle1_y += 1
        elif key == 'i' and self.paddle2_y > 0:
            self.paddle2_y -= 1
        elif key == 'k' and self.paddle2_y < self.height - self.paddle_size:
            self.paddle2_y += 1
            
    def run(self):
        # Set terminal to non-blocking mode
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        try:
            print("Starting Pong Game! Press any key to begin...")
            sys.stdin.read(1)
            
            while True:
                self.draw()
                
                # Get input
                key = self.get_key()
                if key == 'q':
                    break
                    
                if key:
                    self.move_paddles(key)
                
                self.update_ball()
                time.sleep(0.1)  # Game speed
                
        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.clear_screen()
            print(f"Game Over! Final Score - Player 1: {self.score1}, Player 2: {self.score2}")

if __name__ == "__main__":
    game = Pong()
    game.run()