#date: 2022-03-11T16:48:12Z
#url: https://api.github.com/gists/bb332a7952f435b8b9c8f1fb35fb05da
#owner: https://api.github.com/users/pvcraven

import arcade
import math
from dataclasses import dataclass
from typing import Callable
from typing import Tuple

SPRITE_SCALING = 0.5

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Easing Example"


@dataclass
class EasingData:
    start_period: float
    cur_period: float
    end_period: float
    start_value: float
    end_value: float
    ease_function: Callable


def linear(percent: float) -> float:
    return percent


def flip(percent: float) -> float:
    return 1.0 - percent


def ease_in(percent: float) -> float:
    return percent * percent


def ease_out(percent: float) -> float:
    return flip(flip(percent) * flip(percent))


def smooth_step(percent: float) -> float:
    percent = percent * percent * (3.0 - 2.0 * percent)
    return percent


def easing(percent: float, easing_data: EasingData) -> float:
    return easing_data.start_value + (easing_data.end_value - easing_data.start_value) * \
           easing_data.ease_function(percent)


def get_angle_degrees(p1, p2):
    # Position the bullet at the player's current location
    start_x, start_y = p1

    # Get from the mouse the destination location for the bullet
    # IMPORTANT! If you have a scrolling screen, you will also need
    # to add in self.view_bottom and self.view_left.
    dest_x, dest_y = p2

    # Do math to calculate how to get the bullet to the destination.
    # Calculation the angle in radians between the start points
    # and end points. This is the angle the bullet will travel.
    x_diff = dest_x - start_x
    y_diff = dest_y - start_y
    angle = math.degrees(math.atan2(x_diff, y_diff))
    return angle


def ease_angle(start_angle, end_angle, *, time=None, rate=None, ease_function=linear):

    while start_angle - end_angle > 180:
        end_angle += 360

    while start_angle - end_angle < -180:
        end_angle -= 360

    diff = abs(start_angle - end_angle)
    if diff == 0:
        return None

    if rate is not None:
        time = diff / rate

    easing_data = EasingData(start_value=start_angle,
                             end_value=end_angle,
                             start_period=0,
                             cur_period=0,
                             end_period=time,
                             ease_function=ease_function)
    return easing_data


def ease_angle_update(easing_data: EasingData, delta_time: float) -> Tuple:
    done = False
    easing_data.cur_period += delta_time
    if easing_data.cur_period >= easing_data.end_period:
        easing_data.cur_period = easing_data.end_period

    percent = easing_data.cur_period / easing_data.end_period

    angle = easing(percent, easing_data)

    if percent >= 1.0:
        done = True

        while angle > 360:
            angle -= 360

        while angle < 0:
            angle += 360

    return done, angle


def easing_position(start_position, end_position, *, time=None, rate=None, ease_function=linear):
    distance = arcade.get_distance(start_position[0],
                                   start_position[1],
                                   end_position[0],
                                   end_position[1])

    if rate is not None:
        time = distance / rate

    easing_data_x = EasingData(start_value=start_position[0],
                               end_value=end_position[0],
                               start_period=0,
                               cur_period=0,
                               end_period=time,
                               ease_function=ease_function)

    easing_data_y = EasingData(start_value=start_position[1],
                               end_value=end_position[1],
                               start_period=0,
                               cur_period=0,
                               end_period=time,
                               ease_function=ease_function)

    return easing_data_x, easing_data_y


def ease_update(easing_data: EasingData, delta_time: float) -> Tuple:
    done = False
    easing_data.cur_period += delta_time
    if easing_data.cur_period >= easing_data.end_period:
        easing_data.cur_period = easing_data.end_period

    if easing_data.end_period == 0:
        percent = 1.0
        value = easing_data.end_value
    else:
        percent = easing_data.cur_period / easing_data.end_period
        value = easing(percent, easing_data)

    if percent >= 1.0:
        done = True

    return done, value


class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, image, scale):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)

        self.easing_angle_data = None
        self.easing_x_data = None
        self.easing_y_data = None

    def on_update(self, delta_time: float = 1 / 60):
        if self.easing_angle_data is not None:
            done, self.angle = ease_angle_update(self.easing_angle_data, delta_time)
            if done:
                self.easing_angle_data = None

        if self.easing_x_data is not None:
            done, self.center_x = ease_update(self.easing_x_data, delta_time)
            if done:
                self.easing_x_data = None

        if self.easing_y_data is not None:
            done, self.center_y = ease_update(self.easing_y_data, delta_time)
            if done:
                self.easing_y_data = None

    def face_point(self, point: arcade.Point):
        angle = get_angle_degrees(self.position, point)

        # Reverse angle because sprite angles are backwards
        self.angle = -angle


class MyGame(arcade.Window):
    """ Main application class. """

    def __init__(self, width, height, title):
        """ Initializer """

        # Call the parent class initializer
        super().__init__(width, height, title)

        # Variables that will hold sprite lists
        self.player_list = None

        # Set up the player info
        self.player_sprite = None

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)
        self.text = "Test"

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()

        # Set up the player
        self.player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png",
                                    SPRITE_SCALING)
        self.player_sprite.angle = 0
        self.player_sprite.center_x = SCREEN_WIDTH / 2
        self.player_sprite.center_y = SCREEN_HEIGHT / 2
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        """ Render the screen. """

        # This command has to happen before we start drawing
        self.clear()

        # Draw all the sprites.
        self.player_list.draw()

        arcade.draw_text(self.text, 10, 10, arcade.color.WHITE, 18)

    def on_update(self, delta_time):
        """ Movement and game logic """

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        self.player_list.on_update(delta_time)

    def on_key_press(self, key, modifiers):
        x = self.mouse["x"]
        y = self.mouse["y"]

        if key == arcade.key.KEY_1:
            self.player_sprite.face_towards(x, y)
            self.text = "Instant angle change"
        if key in [arcade.key.KEY_2, arcade.key.KEY_3, arcade.key.KEY_4, arcade.key.KEY_5]:
            p1 = self.player_sprite.position
            p2 = (x, y)
            end_angle = -get_angle_degrees(p1, p2)
            start_angle = self.player_sprite.angle
            if key == arcade.key.KEY_2:
                ease_function = linear
                self.text = "Linear easing - angle"
            elif key == arcade.key.KEY_3:
                ease_function = ease_in
                self.text = "Ease in - angle"
            elif key == arcade.key.KEY_4:
                ease_function = ease_out
                self.text = "Ease out - angle"
            elif key == arcade.key.KEY_5:
                ease_function = smooth_step
                self.text = "Smoothstep - angle"
            else:
                raise ValueError("?")

            self.player_sprite.easing_angle_data = ease_angle(start_angle,
                                                              end_angle,
                                                              rate=180,
                                                              ease_function=ease_function)

        if key in [arcade.key.KEY_6, arcade.key.KEY_7, arcade.key.KEY_8, arcade.key.KEY_9]:
            p1 = self.player_sprite.position
            p2 = (x, y)
            if key == arcade.key.KEY_6:
                ease_function = linear
                self.text = "Linear easing - position"
            elif key == arcade.key.KEY_7:
                ease_function = ease_in
                self.text = "Ease in - position"
            elif key == arcade.key.KEY_8:
                ease_function = ease_out
                self.text = "Ease out - position"
            elif key == arcade.key.KEY_9:
                ease_function = smooth_step
                self.text = "Smoothstep - position"
            else:
                raise ValueError("?")

            ex, ey = easing_position(p1, p2, rate=180, ease_function=ease_function)
            self.player_sprite.easing_x_data = ex
            self.player_sprite.easing_y_data = ey

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        self.player_sprite.face_towards(x, y)


def main():
    """ Main function """
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
