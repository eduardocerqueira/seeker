#date: 2021-12-31T17:06:06Z
#url: https://api.github.com/gists/6584389f3bc0eed83f7e79e710878546
#owner: https://api.github.com/users/protopopov1122

#!/usr/bin/env python3
# Simplistic and messy Pacman game implementation for TTY
# Author: Jevgenijs Protopopovs
# License: WTFPL <http://www.wtfpl.net/>
# For more additional madness this can be hosted online via socat:
#     socat tcp-listen:$port,reuseaddr exec:./pacman.py,pty # on server $host
#     socat stdio,rawer tcp:$host:$port                     # on client

import sys
import time
import tty
import termios
import math
import random
from enum import Enum

INIT_MAP = """
###################
#........#........#
#*##.###.#.###.##*#
#.##.###.#.###.##.#
#.................#
#.##.#.#####.#.##.#
#....#...#...#....#
####.### # ###.####
   #.#   &   #.#   
####.# $$A$$ #.####
T   .  $BCD$  .   T
####.# $$$$$ #.####
   #.#       #.#   
####.# ##### #.####
#........#........#
#.##.###.#.###.##.#
#*.#...  @  ...#.*#
##.#.#.#####.#.#.##
#....#...#...#....#
#.######.#.######.#
#.................#
################### 
"""

class Element(Enum):
    Empty = ' '
    Wall = '#'
    House = '$'
    HouseExit = '&'
    Food = '.'
    Energizer = '*'
    Teleport = 'T'
    Pacman = '@'
    Blinky = 'A'
    Pinky = 'B'
    Inky = 'C'
    Clyde = 'D'

CharacterElements = [
    Element.Pacman,
    Element.Blinky,
    Element.Pinky,
    Element.Inky,
    Element.Clyde
]

Ghosts = [
    Element.Blinky,
    Element.Pinky,
    Element.Inky,
    Element.Clyde
]

class Direction(Enum):
    Idle = 0
    Up = 1
    Down = 3
    Left = 2
    Right = 4

    def opposite(self):
        if self == Direction.Idle:
            return None
        elif self == Direction.Up:
            return Direction.Down
        elif self == Direction.Down:
            return Direction.Up
        elif self == Direction.Left:
            return Direction.Right
        elif self == Direction.Right:
            return Direction.Left

class KeyCode(Enum):
    Up = 'w'
    Down = 's'
    Left = 'a'
    Right = 'd'
    Quit = 'q'

class EventType(Enum):
    Keyboard = 0
    MoveCharacter = 1
    ConsumeFood = 2
    ConsumeEnergy = 3
    Death = 4
    GameOver = 5
    Victory = 6

class GhostMode(Enum):
    Scatter = 'S'
    Chase = 'C'
    Frightened = 'F'

class Event:
    def __init__(self, evt_type, data):
        self.type = evt_type
        self.data = data

class EventSource:
    def __init__(self):
        self.event_listeners = list()

    def subscribe(self, listener):
        self.event_listeners.append(listener)

    def publish(self, event):
        for listener in self.event_listeners:
            listener(event)

class Character:
    def __init__(self, map, element, location):
        self.map = map
        self.element = element
        self.location = location

class Map:
    def __init__(self, initial_map):
        map_cells = [list(row) for row in initial_map.strip().split('\n')]
        self.width = None
        self.height = len(map_cells)
        self.map = list()
        self.characters = dict()
        self.ghost_house = set()
        self.ghost_house_exit = None
        self.ghost_home = dict()
        self.pacman_home = None
        self.food = 0
        self.teleport = None
        for y, cell_row in enumerate(map_cells):
            if self.width is None:
                self.width = len(cell_row)
            elif self.width != len(cell_row):
                raise 'Malformed map'
            element_row = list()
            for x, cell in enumerate(cell_row):
                elt = Element(cell)
                if elt in Ghosts:
                    self.ghost_house.add((x, y))
                    self.ghost_home[elt] = (x, y)
                elif elt == Element.House:
                    self.ghost_house.add((x, y))
                    elt = Element.Wall
                elif elt == Element.HouseExit:
                    self.ghost_house_exit = (x, y)
                    elt = Element.Empty
                elif elt == Element.Teleport:
                    if self.teleport is None:
                        self.teleport = ((x, y), None)
                    else:
                        self.teleport = (self.teleport[0], (x, y))
                    elt = Element.Empty
                elif elt == Element.Pacman:
                    self.pacman_home = (x, y)
                elif elt == Element.Food:
                    self.food += 1
                
                if elt in CharacterElements:
                    char = Character(self, elt, (x, y))
                    self.characters[elt] = char
                    element_row.append(Element.Empty)
                else:
                    element_row.append(elt)
            self.map.append(element_row)
    
    def base_at(self, x, y):
        return self.map[y][x]

    def set_base_at(self, x, y, elt):
        self.map[y][x] = elt

    def at(self, x, y):
        for char in self.characters.values():
            if char.location[0] == x and char.location[1] == y:
                return char.element
        return self.base_at(x, y)

class Stats:
    def __init__(self):
        self.points = 0

class Gameplay(EventSource):
    def __init__(self, initial_map):
        EventSource.__init__(self)
        self.map = Map(initial_map)
        self.stats = Stats()
        self.active = True
        self.power_mode = False
        self.power_mode_expires = None
        self.lives = 3

    def is_active(self):
        return self.active

    def stop(self):
        self.active = False

    def move_character(self, character, new_location):
        if new_location[0] < 0 or new_location[1] < 0 or \
            new_location[0] >= self.map.width or new_location[1] >= self.map.height:
            return
        elt = self.map.at(new_location[0], new_location[1])
        old_location = character.location
        if elt == Element.Empty or (elt != Element.Wall and elt != Element.Pacman and character.element in Ghosts):
            character.location = new_location
            self.publish(Event(EventType.MoveCharacter, (character, old_location)))
        elif (character.element == Element.Pacman and elt in Ghosts) or \
                (elt == Element.Pacman and character.element in Ghosts):
            character.location = new_location
            if self.power_mode:
                if elt in Ghosts:
                    self.map.characters[elt].location = self.map.ghost_home[elt]
                    self.publish(Event(EventType.MoveCharacter, (self.map.characters[elt], old_location)))
                    self.publish(Event(EventType.MoveCharacter, (character, old_location)))
                else:
                    character.location = self.map.ghost_home[character.element]
                    self.publish(Event(EventType.MoveCharacter, (character, old_location)))
            elif self.lives > 1:
                self.lives -= 1
                self.publish(Event(EventType.MoveCharacter, (character, old_location)))
                self.publish(Event(EventType.Death, None))
            else:
                self.publish(Event(EventType.GameOver, None))
        elif character.element == Element.Pacman and elt == Element.Food:
            self.stats.points += 1
            self.map.set_base_at(new_location[0], new_location[1], Element.Empty)
            character.location = new_location
            if self.stats.points < self.map.food:
                self.publish(Event(EventType.ConsumeFood, new_location))
            else:
                self.publish(Event(EventType.Victory, None))
            self.publish(Event(EventType.MoveCharacter, (character, old_location)))
        elif character.element == Element.Pacman and elt == Element.Energizer:
            self.map.set_base_at(new_location[0], new_location[1], Element.Empty)
            character.location = new_location
            self.power_mode = True
            self.power_mode_expires = time.time() + 10
            self.publish(Event(EventType.ConsumeEnergy, new_location))
            self.publish(Event(EventType.MoveCharacter, (character, old_location)))

    def run(self):
        if self.power_mode and time.time() > self.power_mode_expires:
            self.power_mode = False

class Console:
    def __init__(self, output):
        self.output = output

    def clear_screen(self):
        self.output.write('\x1b[2J')

    def clear_line(self):
        self.output.write('\x1b[0K')
    
    def reset_cursor(self):
        self.output.write('\x1b[H')

    def reset(self):
        self.clear_screen()
        self.reset_cursor()
    
    def write(self, chars):
        self.output.write(chars)
    
    def newline(self):
        self.write('\r\n')

    def set_position(self, x, y):
        self.write(f'\x1b[{y};{x}H')

    def flush(self):
        self.output.flush()

class Renderer:
    def __init__(self, console, gameplay):
        self.console = console
        self.gameplay = gameplay
        self.events = list()
        self.gameplay.subscribe(lambda evt: self.events.append(evt))

    def finalize_render(self):
        self.console.set_position(0, self.gameplay.map.height + 1)
        self.console.clear_line()
        power_mode = '!' if self.gameplay.power_mode else ' '
        lives = '*' * self.gameplay.lives
        self.console.write(f'{power_mode}{self.gameplay.stats.points}/{self.gameplay.map.food} {lives}')
        self.console.flush()

    def render_base(self):
        self.console.reset()
        self.console.set_position(1, 1)
        for y in range(self.gameplay.map.height):
            for x in range(self.gameplay.map.width):
                elt = self.gameplay.map.base_at(x, y)
                pad = ' '
                if elt == Element.Wall:
                    pad = elt.value
                self.console.write(f'{pad}{elt.value}{pad}')
            self.console.newline()
        for character in self.gameplay.map.characters.values():
            self.console.set_position((character.location[0] + 1) * 3 - 1, character.location[1] + 1)
            self.console.write(character.element.value)
        self.finalize_render()

    def render(self):
        for event in self.events:
            if event.type == EventType.MoveCharacter:
                (character, previous_location) = event.data
                self.console.set_position((previous_location[0] + 1) * 3 - 1, previous_location[1] + 1)
                self.console.write(self.gameplay.map.at(previous_location[0], previous_location[1]).value)
                self.console.set_position((character.location[0] + 1) * 3 - 1, character.location[1] + 1)
                self.console.write(character.element.value)
            elif event.type == EventType.ConsumeFood or event.type == EventType.ConsumeEnergy:
                self.console.set_position((event.data[0] + 1) * 3 - 1, event.data[1] + 1)
                self.console.write(Element.Empty.value)

        self.events = list()
        self.finalize_render()

class InputController(EventSource):
    def __init__(self, input):
        EventSource.__init__(self)
        self.input = input
        self.old_settings = old_settings = termios.tcgetattr(self.input.fileno())
        new_settings = termios.tcgetattr(self.input.fileno())
        new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
        new_settings[6][termios.VMIN] = 0
        new_settings[6][termios.VTIME] = 0
        tty.setraw(self.input)
        termios.tcsetattr(self.input.fileno(), termios.TCSADRAIN, new_settings)

    def run(self):
        while True:
            data = self.input.read()
            if data is None or len(data) == 0:
                break
            for char in data:
                if char == 'q':
                    self.publish(Event(EventType.Keyboard, KeyCode.Quit))
                elif char == 'w':
                    self.publish(Event(EventType.Keyboard, KeyCode.Up))
                elif char == 'a':
                    self.publish(Event(EventType.Keyboard, KeyCode.Left))
                elif char == 's':
                    self.publish(Event(EventType.Keyboard, KeyCode.Down))
                elif char == 'd':
                    self.publish(Event(EventType.Keyboard, KeyCode.Right))

    def close(self):
        termios.tcsetattr(self.input.fileno(), termios.TCSADRAIN, self.old_settings)

def update_location_by_direction(x, y, direction):
    if direction == Direction.Up:
        y -= 1
    elif direction == Direction.Down:
        y += 1
    elif direction == Direction.Left:
        x -= 1
    elif direction == Direction.Right:
        x += 1
    return (x, y)

class PacmanController:
    def __init__(self, input, gameplay):
        self.input = input
        self.gameplay = gameplay
        self.input.subscribe(self.listen)
        self.gameplay.subscribe(self.listen)
        self.direction = Direction.Idle
        self.death = False

    def listen(self, evt):
        if evt.type == EventType.Keyboard:
            if evt.data == KeyCode.Up:
                self.direction = Direction.Up
            elif evt.data == KeyCode.Down:
                self.direction = Direction.Down
            elif evt.data == KeyCode.Right:
                self.direction = Direction.Right
            elif evt.data == KeyCode.Left:
                self.direction = Direction.Left
        elif evt.type == EventType.Death:
            self.death = True

    def run(self):
        char = self.gameplay.map.characters[Element.Pacman]
        if self.death:
            self.death = False
            self.direction = Direction.Idle
            self.gameplay.move_character(char, self.gameplay.map.pacman_home)
            return
        (x, y) = char.location
        if self.direction != Direction.Idle:
            (x, y) = update_location_by_direction(x, y, self.direction)
            if x == self.gameplay.map.teleport[0][0] and y == self.gameplay.map.teleport[0][1]:
                (x, y) = self.gameplay.map.teleport[1]
            elif x == self.gameplay.map.teleport[1][0] and y == self.gameplay.map.teleport[1][1]:
                (x, y) = self.gameplay.map.teleport[0]
            self.gameplay.move_character(char, (x, y))

    def close(self):
        pass

class GhostController:
    def __init__(self, element, pacman, speed, gameplay):
        self.element = element
        self.pacman = pacman
        self.speed = speed
        self.gameplay = gameplay
        self.mode = None
        self.next_mode_change = None
        self.direction = Direction.Idle
        self.beat = 0
        self.gameplay.subscribe(self.listen)
        self.death = False

    def listen(self, evt):
        if evt.type == EventType.Death:
            self.death = True

    def run(self):
        if self.death:
            self.death = False
            self.beat = 0
            self.direction = Direction.Idle
            home = self.gameplay.map.ghost_home[self.element]
            self.gameplay.move_character(self.gameplay.map.characters[self.element], home)
            self.mode = None
            self.next_mode_change = None
            return
        if self.mode == None:
            self.next_mode_change = time.time() + 5
            self.mode = GhostMode.Scatter
        elif time.time() > self.next_mode_change:
            self.mode = GhostMode.Chase
        if self.mode == GhostMode.Chase and not self.gameplay.power_mode:
            self.chase()
        elif self.mode == GhostMode.Scatter and not self.gameplay.power_mode:
            self.scatter()
        else:
            self.frightened()
        
        char = self.gameplay.map.characters[self.element]
        (x, y) = char.location
        if self.direction != Direction.Idle and self.beat % self.speed != 0:
            (x, y) = update_location_by_direction(x, y, self.direction)
            self.gameplay.move_character(char, (x, y))
        self.beat = (self.beat + 1) % self.speed

    def chase(self):
        if self.element == Element.Blinky:
            (target_x, target_y) = self.gameplay.map.characters[Element.Pacman].location
        elif self.element == Element.Pinky:
            (target_x, target_y) = self.gameplay.map.characters[Element.Pacman].location
            target_x, target_y = update_location_by_direction(target_x, target_y, self.pacman.direction)
            target_x, target_y = update_location_by_direction(target_x, target_y, self.pacman.direction)
            target_x, target_y = update_location_by_direction(target_x, target_y, self.pacman.direction)
            target_x, target_y = update_location_by_direction(target_x, target_y, self.pacman.direction)
        elif self.element == Element.Inky:
            (pacman_x, pacman_y) = self.gameplay.map.characters[Element.Pacman].location
            pacman_x, pacman_x = update_location_by_direction(pacman_x, pacman_x, self.pacman.direction)
            pacman_x, pacman_x = update_location_by_direction(pacman_x, pacman_x, self.pacman.direction)
            (blinky_x, blinky_y) = self.gameplay.map.characters[Element.Blinky].location
            (diff_x, diff_y) = pacman_x - blinky_x, pacman_y - blinky_y
            target_x, target_y = blinky_x + diff_x * 2, blinky_y + diff_y * 2
        elif self.element == Element.Clyde:
            (pacman_x, pacman_y) = self.gameplay.map.characters[Element.Pacman].location
            (clyde_x, clyde_y) = self.gameplay.map.characters[Element.Clyde].location
            distance = math.sqrt((pacman_x - clyde_x)**2 + (pacman_y - clyde_y)**2)
            if distance < 8:
                target_x, target_y = 0, self.gameplay.map.height
            else:
                target_x, target_y = pacman_x, pacman_y
        self.navigate(target_x, target_y)

    def scatter(self):
        current_loc = self.gameplay.map.characters[self.element].location
        if current_loc in self.gameplay.map.ghost_house:
            target_x, target_y = self.gameplay.map.ghost_house_exit
        elif self.element == Element.Blinky:
            target_x, target_y = self.gameplay.map.width, 0
        elif self.element == Element.Pinky:
            target_x, target_y = 0, 0
        elif self.element == Element.Inky:
            target_x, target_y = self.gameplay.map.width, self.gameplay.map.height
        elif self.element == Element.Clyde:
            target_x, target_y = 0, self.gameplay.map.height
        self.navigate(target_x, target_y)

    def frightened(self):
        directions = self.possible_directions()
        if directions:
            self.direction = random.choice(directions)[2]

    def navigate(self, target_x, target_y):
        directions = self.possible_directions()
        if directions is None:
            return
        min_distance = math.inf
        new_direction = self.direction
        for next_x, next_y, next_direction in directions:
            distance = (target_x - next_x)**2 + (target_y - next_y) ** 2
            if distance < min_distance and (next_direction != self.direction.opposite() or len(directions) == 1):
                new_direction = next_direction
                min_distance = distance
            elif distance == min_distance and next_direction != self.direction.opposite() and next_direction.value < new_direction.value:
                new_direction = next_direction
                min_distance = distance
        self.direction = new_direction

    def possible_directions(self):
        directions = []
        (x, y) = self.gameplay.map.characters[self.element].location
        if x - 1 >= 0 and self.gameplay.map.base_at(x - 1, y) != Element.Wall:
            directions.append((x - 1, y, Direction.Left))
        if x + 1 < self.gameplay.map.width and self.gameplay.map.base_at(x + 1, y) != Element.Wall:
            directions.append((x + 1, y, Direction.Right))
        if y - 1 >= 0 and self.gameplay.map.base_at(x, y - 1) != Element.Wall:
            directions.append((x, y - 1, Direction.Up))
        if y + 1 < self.gameplay.map.height and self.gameplay.map.base_at(x, y + 1) != Element.Wall:
            directions.append((x, y + 1, Direction.Down))
        if (x, y) not in self.gameplay.map.ghost_house:
            directions = [(x, y, direction) for x, y, direction in directions if (x, y) not in self.gameplay.map.ghost_house]
        current_direction_valid = any([direction == self.direction for _, _, direction in directions])
        if (len(directions) < 2 and self.direction != Direction.Idle and current_direction_valid) or len(directions) == 0:
            return None # Not an intersection
        return directions

    def close(self):
        pass

class GameLoop:
    def __init__(self, renderer, controllers, gameplay):
        self.renderer = renderer
        self.controllers = controllers
        self.gameplay = gameplay

    def run(self):
        self.renderer.render_base()
        while self.gameplay.is_active():
            for ctrl in self.controllers:
                ctrl.run()
            self.gameplay.run()
            self.renderer.render()
            time.sleep(0.25)

def pacman():
    gameplay = Gameplay(INIT_MAP)
    console = Console(sys.stdout)
    renderer = Renderer(console, gameplay)
    input_ctrl = InputController(sys.stdin)
    pacman_ctrl = PacmanController(input_ctrl, gameplay)
    controllers = [
        input_ctrl,
        pacman_ctrl,
        GhostController(Element.Blinky, pacman_ctrl, 7, gameplay),
        GhostController(Element.Pinky, pacman_ctrl, 7, gameplay),
        GhostController(Element.Inky, pacman_ctrl, 7, gameplay),
        GhostController(Element.Clyde, pacman_ctrl, 7, gameplay)
    ]
    game = GameLoop(renderer, controllers, gameplay)
    def quit_callback(evt):
        if (evt.type == EventType.Keyboard and evt.data == KeyCode.Quit) or \
            evt.type == EventType.GameOver or \
            evt.type == EventType.Victory:
            gameplay.stop()
    input_ctrl.subscribe(quit_callback)
    gameplay.subscribe(quit_callback)
    game.run()
    for ctrl in controllers:
        ctrl.close()

if __name__ == '__main__':
    pacman()
