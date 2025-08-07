#date: 2025-08-07T17:08:35Z
#url: https://api.github.com/gists/856736665eecb0f02cf0a4e7e9a76eca
#owner: https://api.github.com/users/EncodeTheCode

import time
import sys
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *

global timer, timer2
timer, timer2 = False, False
global display_w, display_h
display_w = 1920
display_h = 1080

# --- Weapon & HUD Configuration ---
gfx  = "gfx/"
itm2 = gfx + "items/sf2/"
wpn  = gfx + "wpn/sf/"
wpn2 = gfx + "wpn/sf2/"
ext  = ".png"
sfx = 'sfx/'
wsfx = sfx + 'wpn/'
xs = '.mp3'

# --- Fixed click delay (seconds) ---
click_delay = 0.05  # 50 ms

configs = [
    {'id':  0, 'name':  'Knife',                  'path': wpn2 + 'knife_sf2'            + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'knife' + xs, 'fire_rate': 2.000},    # Melee, no auto fire
    {'id':  1, 'name':  'Hand Taser',             'path': wpn2 + 'hand_taser_sf2'       + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 2.000},
    {'id':  2, 'name':  'Air Taser',              'path': wpn  + 'air_taser_sf'         + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 2.000},
    {'id':  3, 'name':  'Crossbow',               'path': wpn2 + 'crossbow_sf2'         + ext, 'max_primary': 5,  'max_secondary':  None, 'sfx_path': wsfx + 'crossbow' + xs,  'fire_rate': 0.75},
    {'id':  4, 'name':  '9MM',                    'path': wpn2 + '9mm_sf2'              + ext, 'max_primary': 15,  'max_secondary': 75, 'sfx_path': wsfx + '9mm' + xs,  'fire_rate': 0.150},
    {'id':  5, 'name':  '9MM Silenced',           'path': wpn2 + '9mm_silenced_sf2'     + ext, 'max_primary': 15,  'max_secondary': 75, 'sfx_path': wsfx + '9mm_silenced' + xs,  'fire_rate': 0.150},
    {'id':  6, 'name':  '.45',                    'path': wpn2 + '45_sf2'               + ext, 'max_primary': 10,  'max_secondary': 50, 'sfx_path': wsfx + 'collect' + xs,  'fire_rate': ((0.400 * 2) + click_delay)},
    {'id':  7, 'name':  'G-18',                   'path': wpn2 + 'g18_sf2'              + ext, 'max_primary': 33,  'max_secondary': 165, 'sfx_path': wsfx + 'collect' + xs,  'fire_rate': 0.050},
    {'id':  8, 'name':  'HK5',                    'path': wpn2 + 'hk5_sf2'              + ext, 'max_primary': 32,  'max_secondary': 160, 'sfx_path': wsfx + 'hk5' + xs,  'fire_rate': (0.071 * 2) - click_delay},
    {'id':  9, 'name':  'HK5 Silenced',           'path': wpn2 + 'hk5_silenced_sf2'     + ext, 'max_primary': 32,  'max_secondary': 160, 'sfx_path': wsfx + 'hk5' + xs,  'fire_rate': (0.071 * 2) - click_delay},
    {'id': 10, 'name':  'K3G4',                   'path': wpn2 + 'k3g4_sf2'             + ext, 'max_primary': 20,  'max_secondary': 100, 'sfx_path': wsfx + 'smg_gen01' + xs,  'fire_rate': 0.071},
    {'id': 11, 'name':  'BIZ-2',                  'path': wpn2 + 'biz2_sf2'             + ext, 'max_primary': 66,  'max_secondary': 330, 'sfx_path': wsfx + 'smg_gen02' + xs,  'fire_rate': 0.071},
    {'id': 12, 'name':  'M-16',                   'path': wpn2 + 'm16_sf2'              + ext, 'max_primary': 30,  'max_secondary': 150, 'sfx_path': wsfx + 'm16' + xs, 'fire_rate': 0.0715},
    {'id': 13, 'name':  'Shotgun',                'path': wpn2 + 'shotgun_sf2'          + ext, 'max_primary': 25,  'max_secondary': None, 'sfx_path': wsfx + 'shotgun' + xs, 'fire_rate': ((0.400 * 2) + click_delay)},
    {'id': 14, 'name':  'UAS-12',                 'path': wpn2 + 'uas12_sf2'            + ext, 'max_primary': 12,  'max_secondary': None, 'sfx_path': wsfx + 'uas12' + xs,  'fire_rate': 0.150},
    {'id': 15, 'name':  'Sniper Rifle',           'path': wpn2 + 'sniper_rifle_unsilenced_sf2' + ext, 'max_primary': 10, 'max_secondary': 20, 'sfx_path': wsfx + 'old_sniper_rifle' + xs, 'fire_rate': ((0.400 * 2) + click_delay)},
    {'id': 16, 'name':  'Sniper Rifle Silenced',  'path': wpn2 + 'sniper_rifle_sf2'     + ext, 'max_primary': 10,  'max_secondary': 20, 'sfx_path': wsfx + 'old_sniper_rifle' + xs, 'fire_rate': ((0.400 * 2) + click_delay)},
    {'id': 17, 'name':  'Old Sniper Rifle',       'path': wpn  + 'sniper_rifle_sf1'     + ext, 'max_primary': 10,  'max_secondary': 20, 'sfx_path': wsfx + 'old_sniper_rifle' + xs, 'fire_rate': ((0.400 * 2) + click_delay)},
    {'id': 18, 'name':  'Dragunov',               'path': wpn2 + 'dragunov_sf2'         + ext, 'max_primary': 10,  'max_secondary': 20, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 0.2},
    {'id': 19, 'name':  'M-79',                   'path': wpn2 + 'm79_sf2'              + ext, 'max_primary': 15,  'max_secondary': None, 'sfx_path': wsfx + 'launcher' + xs, 'fire_rate': ((0.2 * 2) + click_delay)},
    {'id': 20, 'name':  'H11',                    'path': wpn2 + 'h11_sf2'              + ext, 'max_primary': 50,  'max_secondary': 250, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 0.035},
    {'id': 21, 'name':  'Grenade',                'path': wpn2 + 'grenade_sf2'          + ext, 'max_primary': 10,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 1.0},
    {'id': 22, 'name':  'Gas Grenade',            'path': wpn2 + 'gas_grenade_sf2'      + ext, 'max_primary': 10,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 1.0},
    {'id': 23, 'name':  'Flamethrower',           'path': wpn2 + 'flamethrower_sf2'     + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 2.000},
    {'id': 24, 'name':  'Keycard',                'path': itm2 + 'card_sf2'             + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 0.0},
    {'id': 25, 'name':  'Binoculars',             'path': itm2 + 'binoculars_sf'        + ext, 'max_primary': -1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 0.0},
    {'id': 26, 'name':  'C4',                     'path': itm2 + 'c4_sf2'               + ext, 'max_primary':  1,  'max_secondary': None, 'sfx_path': wsfx + 'collect' + xs, 'fire_rate': 1.0},
]

# --- Play sound after time elapsed ---
def play_sound_after_delay(file_path, delay_ms=0, volume=1.0):
    """
    Waits delay_ms milliseconds (supports fractional ms), then plays sound.
    Plays immediately if delay_ms <= 0.
    
    Args:
      file_path (str): Path to the sound file.
      delay_ms (float): Delay in milliseconds (can be fractional).
      volume (float): Volume (0.0 to 1.0).
    """
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)  # Sleep fractional milliseconds
    
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")

class Timer:
    def __init__(self, delay_ms, callback):
        """
        delay_ms: milliseconds to wait before calling callback
        callback: function to call when timer expires
        """
        self.delay = delay_ms / 1000.0
        self.callback = callback
        self.start_time = time.perf_counter()
        self.finished = False

    def update(self):
        """
        Call this frequently (e.g. every frame).
        Calls callback once delay has elapsed.
        """
        if self.finished:
            return
        now = time.perf_counter()
        if now - self.start_time >= self.delay:
            self.callback()
            self.finished = True

# --- Weapon System ---
class Weapon:
    __slots__ = ('id', 'name', 'primary', 'secondary',
                 'max_primary', 'max_secondary', 'reloading')

    def __init__(self, cfg):
        self.id = cfg['id']
        self.name = cfg['name']
        self.max_primary = cfg['max_primary']
        self.max_secondary = cfg.get('max_secondary') or 0
        self.primary = self.max_primary
        self.secondary = self.max_secondary
        self.reloading = False

    def fire_primary(self):
        if not self.reloading and self.primary > 0:
            self.primary -= 1
            return True
        return False

    def start_reload(self, duration_ms=2500):
        if self.reloading or self.primary == self.max_primary or self.secondary == 0:
            return
        self.reloading = True
        pygame.time.set_timer(pygame.USEREVENT + 1, duration_ms)

        w = self.id

        if w in (4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20):
            if self.reloading:
                timer = Timer(25, play_sound_after_delay(f"{wsfx}reload_start{xs}", 0))

    def finish_reload(self):
        needed = self.max_primary - self.primary
        to_load = min(needed, self.secondary)
        self.primary += to_load
        self.secondary -= to_load
        self.reloading = False
        pygame.time.set_timer(pygame.USEREVENT + 1, 0)  # stop timer

        w = self.id
        
        if w in (4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20):
            if self.reloading == False:
                timer2 = Timer(0, play_sound_after_delay(f"{wsfx}reload_end{xs}", 0))

    def cancel_reload(self):
        if self.reloading:
            self.reloading = False
            pygame.time.set_timer(pygame.USEREVENT + 1, 0)  # cancel reload timer


class WeaponSystem:
    __slots__ = ('weapons', 'active')

    def __init__(self, configs):
        self.weapons = {}
        self.active = None
        for cfg in configs:
            w = Weapon(cfg)
            self.weapons[w.name] = w
            if self.active is None:
                self.active = w

    def switch_weapon(self, name):
        # Prevent switching during reload
        if self.active and self.active.reloading:
            return  # Block switching
        self.active = self.weapons.get(name, self.active)

    def fire(self):
        return self.active.fire_primary()

    def reload(self):
        w = self.active
        
        if w.id in (4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20):
            self.active.start_reload()

    def update(self):
        # Call regularly to update reload timers
        if self.active:
            self.active.update()

    def get_hud_lines(self):
        w = self.active
        if w.primary == -1:
            return []

        status = 'Reloading...' if w.reloading else 'Ready'

        if w.id in (3, 13, 14, 19, 21, 22, 26):
            lines = [
                f"{w.primary:02d}", " ",
                f"  "
            ]
        else:
            lines = [
                f"{w.primary:02d} / {w.secondary:02d}", " ",
                f"  "
            ]

        if w.max_secondary > 0:
            ammo_line = f"  {status}"
            lines.append(ammo_line)
        else:
            if w.id in (0, 1, 2, 3, 13, 14, 19, 21, 22, 24, 25, 26):
                lines.append("")
            else:
                lines.append(str(w.primary))

        return lines

# --- HUD Rendering Classes ---
class HUDItem:
    def __init__(self, cfg, speed, screen_w, y_pos):
        assert os.path.isfile(cfg['path']), f"Missing HUD texture: {cfg['path']}"
        surf = pygame.image.load(cfg['path']).convert_alpha()
        self.name = cfg['name']
        self.w, self.h = surf.get_size()
        
        # Generate OpenGL texture
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, pygame.image.tostring(surf, 'RGBA', True)
        )
        
        self.off_x = screen_w + 125
        self.ons_x = lambda: screen_w - 50 - self.w
        self.x = self.target = self.off_x
        self.y = y_pos
        self.speed = speed

    def update(self, dt):
        diff = self.target - self.x
        if abs(diff) > 1:
            step = self.speed * dt * (1 if diff > 0 else -1)
            # Avoid overshoot
            if abs(step) < abs(diff):
                self.x += step
            else:
                self.x = self.target

    def draw(self):
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        for u, v, dx, dy in (
            (0, 0, 0, 0),
            (1, 0, self.w, 0),
            (1, 1, self.w, self.h),
            (0, 1, 0, self.h)
        ):
            glTexCoord2f(u, v)
            glVertex2f(self.x + dx, self.y + dy)
        glEnd()
        glDisable(GL_TEXTURE_2D)

class WeaponHUD:
    def __init__(self, configs, screen_size, y_pos=10, ammo_x=10, ammo_y=100, roll_speed=600.0, switch_delay=250):
        self.W, self.H = screen_size
        self.off_x = self.W + 125
        self.ons_x = lambda itm: self.W - 50 - itm.w
        self.speed = roll_speed
        self.delay = switch_delay
        self.y_pos = y_pos
        self.ammo_x = ammo_x
        self.ammo_y = ammo_y
        self.items = [HUDItem(cfg, self.speed, self.W, y_pos) for cfg in configs]
        for itm in self.items:
            itm.x = itm.target = self.off_x
        self.idx = 0
        self.paused = False
        self.rolling = None
        self.rolled_in = False
        self.last_switch = 0
        self.roll('in')

    def set_ammo_position(self, x, y):
        """Easily reposition the ammo text on the HUD."""
        self.ammo_x = x
        self.ammo_y = y

    def roll(self, direction):
        itm = self.items[self.idx]
        if direction == 'in':
            itm.x = self.off_x
            itm.target = self.ons_x(itm)
            self.rolled_in = False
        else:
            itm.target = self.off_x
        self.rolling = direction

    def switch_weapon(self, delta, wsys):
        now = pygame.time.get_ticks()
        if self.rolling or now - self.last_switch < self.delay:
            return
        # Block switch if weapon is reloading
        if wsys.active and wsys.active.reloading:
            return

        self.last_switch = now
        self.items[self.idx].x = self.off_x
        self.idx = (self.idx + delta) % len(self.items)
        new = self.items[self.idx]
        new.x = new.target = self.ons_x(new)
        self.rolling = None
        self.rolled_in = True
        wsys.switch_weapon(self.items[self.idx].name)

    def update_and_draw(self, dt):
        cur = self.items[self.idx]
        if self.rolling:
            cur.update(dt)
            cur.draw()
            if abs(cur.x - cur.target) < 1:
                self.rolled_in = (cur.target < self.off_x)
                self.rolling = None
        elif not self.paused and self.rolled_in:
            weapon_offsets = {
                "Knife": (222, 55),
                "Crossbow": (282, 55),
                "9MM": (198, 55),
                "9MM Silenced": (198, 55),
                ".45": (210, 55),
                "G-18": (195, 58),
                "HK5": (268, 51),
                "HK5 Silenced": (268, 51),
                "K3G4": (255, 55),
                "BIZ-2": (275, 51),
                "M-16": (232, 51),
                "Shotgun": (315, 55),
                "UAS-12": (225, 51),
                "Sniper Rifle": (247, 53),
                "Sniper Rifle Silenced": (247, 53),
                "Old Sniper Rifle": (245, 51),
                "Dragunov": (275, 51),
                "M-79": (275, 58),
                "H11": (235, 51),
                "Grenade": (152, 50),
                "Gas Grenade": (152, 50),
                "C4": (225, 51),
            }

            x_offset, y_offset = weapon_offsets.get(cur.name, (222, 55))
            self.set_ammo_position(display_w - x_offset, display_h - y_offset)
            cur.draw()
        elif self.paused or self.rolled_in == False or self.rolling == True:
            po=(-420,0)
            weapon_offsets = {n['name']:po for n in configs}

            x_offset, y_offset = weapon_offsets.get(cur.name, (222, 55))
            self.set_ammo_position(display_w - x_offset, display_h - y_offset)
            cur.draw()

    def handle_event(self, event, wsys):
        if event.type == KEYDOWN:
            if event.key == K_p:
                self.paused = not self.paused
                self.roll('in' if not self.paused else 'out')
            elif event.key == K_RIGHT and not self.paused:
                self.switch_weapon(1, wsys)
            elif event.key == K_LEFT and not self.paused:
                self.switch_weapon(-1, wsys)

# --- Main Loop ---
def main():
    # 1. Set audio settings before mixer init
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)

    # 2. Fully initialize Pygame (this includes mixer *if not yet done manually*)
    pygame.init()

    # 3. Explicitly init mixer (optional if already handled via pre_init + init)
    pygame.mixer.init()

    # 4. Set channel count after mixer is definitely initialized
    pygame.mixer.set_num_channels(64)
    
    screen_size = (display_w, display_h)
    screen = pygame.display.set_mode(screen_size, DOUBLEBUF | OPENGL)
    glOrtho(0, screen_size[0], 0, screen_size[1], -1, 1)
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 22)
    white = (255, 255, 255)

    wsys = WeaponSystem(configs)
    hud = WeaponHUD(
        configs,
        screen_size,
        y_pos=50,
        ammo_x=(display_w - 222),
        ammo_y=(display_h - 55),
        roll_speed=700.0,      # Optional: you can omit to use default
        switch_delay=350       # Optional: default delay between switches
    )

    weapon_sounds = {}
    for w in configs:
        if 'sfx_path' in w:
            try:
                weapon_sounds[w['id']] = pygame.mixer.Sound(w['sfx_path'])
            except pygame.error as e:
                print(f"Failed to load sound {w['sfx_path']}: {e}")

    running = True
    mouse_held = False
    last_fire_time = 0.0
    cur_wpn_fire_rate = 0.0  # seconds between shots

    while running:
        dt = clock.tick(60) / 1000.0
        current_time_s = pygame.time.get_ticks() / 1000.0

        if timer:
            try:
                timer.update()
            except:
                pass
        if timer2:
            try:
                timer2.update()
            except:
                pass

        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == USEREVENT + 1:
                wsys.active.finish_reload()
                pygame.time.set_timer(USEREVENT + 1, 0)
            else:
                hud.handle_event(e, wsys)
                if e.type == KEYDOWN and e.key == K_r:
                    if hud.paused == False:
                        wsys.reload()
                    else:
                        pass
                elif e.type == MOUSEBUTTONDOWN and e.button == 1:
                    # Always start tracking hold
                    mouse_held = True
                    # Compute cooldown based on weapon's fire_rate
                    wid = wsys.active.id
                    rate = next((w['fire_rate'] for w in configs if w['id']==wid), 0.0)
                    cooldown = click_delay + rate
                    # Fire if cooldown elapsed
                    if hud.paused == False:
                        if current_time_s - last_fire_time >= cooldown and wsys.fire():
                            last_fire_time = current_time_s
                            if weapon_sounds[wid]:
                                weapon_sounds[wid].play()
                    else:
                        pass

                elif e.type == MOUSEBUTTONUP and e.button == 1:
                    mouse_held = False

        # When game is unpaused
        if hud.paused == False:
            # Auto-fire while holding, only based on fire_rate
            if mouse_held:
                wid = wsys.active.id
                rate = next((w['fire_rate'] for w in configs if w['id']==wid), 0.0)
                # ignore click_delay here
                if rate > 0 and current_time_s - last_fire_time >= rate:
                    if wsys.fire():
                        last_fire_time = current_time_s
                        if weapon_sounds[wid]: weapon_sounds[wid].play()
        else:
            pass

        glClear(GL_COLOR_BUFFER_BIT)
        hud.update_and_draw(dt)

        # Ammo text
        lines = wsys.get_hud_lines()
        for i, ln in enumerate(lines):
            text_surface = font.render(ln, True, white)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2d(hud.ammo_x, screen_size[1] - hud.ammo_y - i * 12)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        pygame.display.flip()
    pygame.quit()

if __name__=='__main__':
    main()
