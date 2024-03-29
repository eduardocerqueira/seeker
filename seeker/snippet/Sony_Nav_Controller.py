#date: 2024-03-29T17:07:16Z
#url: https://api.github.com/gists/56e344bec9d715dd470aaf50a980aff9
#owner: https://api.github.com/users/Locyst

import pygame

class SonyNavController:
    def __init__(self, controller_num=0):
        pygame.init()
        self.joystick = pygame.joystick.Joystick(controller_num)
        self.joystick.init()
        
        self.TOLERANCE = 0.99
        self.x = 0
        self.y = 0

        self.L2_THRESHOLD_HALF = 0.5
        self.L2_THRESHOLD_FULL = 0.9

        self.button_handlers = {
        0: self.x_button_handler,
        1: self.o_button_handler,
        4: self.l1_button_handler,
        5: self.l2_button_handler,
        6: self.play_station_button_handler,
        7: self.joystick_button_handler,
        8: self.up_pad_button_handler,
        9: self.down_pad_button_handler,
        10: self.left_pad_button_handler,
        11: self.right_pad_button_handler,
    }
    
    def x_button_handler(self):
        print("X button pressed")
        
    def o_button_handler(self):
        print("o button pressed")

    def l1_button_handler(self):
        print("l1 button pressed")

    def l2_button_handler(self):
        pass

    def play_station_button_handler(self):
        print("play station button pressed")

    def joystick_button_handler(self):
        print("joystick button pressed")

    def up_pad_button_handler(self):
        print("up pad button pressed")

    def down_pad_button_handler(self):
        print("down pad button pressed")

    def left_pad_button_handler(self):
        print("left pad button pressed")

    def right_pad_button_handler(self):
        print("right pad button pressed")
        
    def down_axis_handler(self):
        print("axis is down")
        
    def up_axis_handler(self):
        print("axis is up")
        
    def right_axis_handler(self):
        print("axis is right")
        
    def left_axis_handler(self):
        print("axis is left")

    def l2_axis_handler(self, pos):
        if pos > self.L2_THRESHOLD_FULL:
            print("L2 fully clicked")
        elif pos > self.L2_THRESHOLD_HALF:
            print("L2 halfway clicked")
        elif pos < -self.L2_THRESHOLD_HALF:
            print("L2 released")
        elif pos < -self.L2_THRESHOLD_FULL:
            print("No longer holding L2")

    def get_input(self):
        try:
            while True:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.JOYBUTTONDOWN:
                        button = event.button
                        if button in self.button_handlers:
                            self.button_handlers[button]()

                    elif event.type == pygame.JOYAXISMOTION:
                        horiz_move = self.joystick.get_axis(0)
                        vert_move = self.joystick.get_axis(1)
                        l2_move = self.joystick.get_axis(2)
                        if abs(vert_move) > self.TOLERANCE:
                            if vert_move > 0:
                                self.down_axis_handler()
                            elif vert_move < 0:
                                self.up_axis_handler()
                        if abs(horiz_move) > self.TOLERANCE:
                            if horiz_move > 0:
                                self.right_axis_handler()
                            elif horiz_move < 0:
                                self.left_axis_handler()
                        if abs(l2_move) > (self.TOLERANCE / 4):
                            self.l2_axis_handler(l2_move)


        except KeyboardInterrupt:
            print("EXITING NOW")
            self.joystick.quit()

SonyNavController().get_input()