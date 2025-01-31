#date: 2025-01-31T17:06:51Z
#url: https://api.github.com/gists/7017bf3056d7358f2fc6260e8aba890d
#owner: https://api.github.com/users/KiNg-m-coder

import keyboard
import pygame

pygame.mixer.init()  
def play_sound():
    pygame.mixer.Sound("death_bed.mp3").play()  

keyboard.add_hotkey("a", play_sound)

print("Press 'A' to play the sound. Press 'Esc' to exit.")
keyboard.wait("esc")  
