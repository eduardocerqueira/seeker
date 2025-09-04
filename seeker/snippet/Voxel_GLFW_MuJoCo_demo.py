#date: 2025-09-04T16:44:03Z
#url: https://api.github.com/gists/45901c7cc2f5e659696d136c71c240a1
#owner: https://api.github.com/users/Niptlox

"""
Voxel Game with MuJoCo & GLFW
A minimal 3D voxel sandbox game using MuJoCo for physics and GLFW for rendering. Features terrain generation with Perlin noise, block breaking/placing, first-person camera, and basic physics. Textures are auto-generated.

Short demo of procedural world interaction in MuJoCo.
"""
import mujoco
import glfw
import numpy as np
import time
from perlin_noise import PerlinNoise
from PIL import Image
import os
import logging
import math

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Определения блоков ---
BLOCK_TYPES = {"air": 0, "grass": 1, "dirt": 2, "stone": 3, "bedrock": 4}
BLOCK_MATERIALS = {1: "mat_grass", 2: "mat_dirt", 3: "mat_stone", 4: "mat_bedrock"}

# --- Генерация мира ---
class VoxelWorld:
    def __init__(self, size=(20, 20, 32)):
        logging.info(f"Инициализация мира размером {size}.")
        self.size = size
        self.blocks = np.zeros(size, dtype=np.int8)
        self.model = None
        self.data = None
        self.generate_terrain()

    def generate_terrain(self):
        logging.info("Начало генерации ландшафта...")
        seed = np.random.randint(0, 1000)
        noise = PerlinNoise(octaves=4, seed=seed)
        logging.info(f"Генерация шума с seed={seed}.")
        
        width, depth, height_max = self.size
        terrain_base_height = height_max // 2 
        terrain_variation = height_max // 4

        for x in range(width):
            for z in range(depth):
                noise_val = noise([x * 0.05, z * 0.05])
                y_terrain = int(noise_val * terrain_variation + terrain_base_height)
                y_terrain = max(0, min(y_terrain, height_max - 1))

                self.blocks[x, z, 0] = BLOCK_TYPES["bedrock"]
                for y in range(1, y_terrain - 3):
                    self.blocks[x, z, y] = BLOCK_TYPES["stone"]
                for y in range(y_terrain - 3, y_terrain):
                    if y > 0: self.blocks[x, z, y] = BLOCK_TYPES["dirt"]
                if y_terrain > 0: self.blocks[x, z, y_terrain] = BLOCK_TYPES["grass"]
        logging.info("Генерация ландшафта завершена.")

    def create_mujoco_model_xml(self):
        logging.debug("Создание XML-строки для модели MuJoCo.")
        player_pos = f"{self.size[0]/2} {self.size[1]/2} {self.size[2] - 10}"
        
        xml_parts = [f"""
<mujoco model="voxel_world">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true" texturedir="textures"/>
    <option integrator="RK4" timestep="0.01"/>
    <visual>
        <headlight ambient="0.6 0.6 0.6" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
        <map znear="0.01"/>
        <quality shadowsize="4096"/>
    </visual>
    <asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grass" type="cube" file="grass.png" />
        <texture name="dirt" type="cube" file="dirt.png" />
        <texture name="stone" type="cube" file="stone.png" />
        <texture name="bedrock" type="cube" file="bedrock.png" />
        <material name="mat_grass" texture="grass" specular="0.1" shininess="0.1" reflectance="0"/>
        <material name="mat_dirt" texture="dirt" specular="0.1" shininess="0.1" reflectance="0"/>
        <material name="mat_stone" texture="stone" specular="0.1" shininess="0.1" reflectance="0"/>
        <material name="mat_bedrock" texture="bedrock" specular="0.1" shininess="0.1" reflectance="0"/>
        <material name="mat_player" rgba="0.8 0.8 1 0.3"/> <!-- полупрозрачный материал -->
    </asset>
    <worldbody>
        <light pos="0 0 100" dir="0 0 -1" directional="true"/>
        """]
        
        active_blocks = 0
        for x in range(self.size[0]):
            for z in range(self.size[1]):
                for y in range(self.size[2]):
                    block_type = self.blocks[x, z, y]
                    if block_type != BLOCK_TYPES["air"]:
                        material = BLOCK_MATERIALS.get(block_type, "")
                        xml_parts.append(f'<geom name="block_{x}_{y}_{z}" type="box" size="0.5 0.5 0.5" pos="{x} {z} {y}" material="{material}" contype="1" conaffinity="1" group="1" friction="0.1 0.1 0.1" />')
                        active_blocks += 1
        
        logging.debug(f"Добавлено {active_blocks} блоков в XML.")
        xml_parts.append(f"""
        <body name="player" pos="{player_pos}">
            <joint name="player_x" type="slide" axis="1 0 0"/>
            <joint name="player_y" type="slide" axis="0 1 0"/>
            <joint name="player_z" type="slide" axis="0 0 1"/>
            <geom name="player_geom" type="cylinder" size="0.4 0.8" mass="80" friction="0.01 0.01 0.01" material="mat_player" contype="1" conaffinity="0"/>
        </body>
    </worldbody>
</mujoco>""")
        return "\n".join(xml_parts)

    def update_model(self):
        logging.info("Генерация XML и компиляция модели MuJoCo...")
        xml_string = self.create_mujoco_model_xml()
        try:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
            self.data = mujoco.MjData(self.model)
            logging.info("Модель и данные MuJoCo успешно обновлены.")
        except Exception as e:
            logging.error(f"Ошибка при обновлении модели MuJoCo: {e}")
            raise

# --- Основной класс игры ---
class VoxelGame:
    def __init__(self):
        logging.info("Создание экземпляра игры VoxelGame.")
        self.world = VoxelWorld()
        self.world.update_model()
        self.model = self.world.model
        self.data = self.world.data
        
        self.move_speed = 250.0
        self.jump_force = 50000.0
        self.key_states = {'W': False, 'A': False, 'S': False, 'D': False, ' ': False}
        self.yaw, self.pitch = -90.0, 0.0
        self.last_x, self.last_y = 640, 360
        self.first_mouse = True
        
        self.window = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = None

        self.break_distance = 5.0  # максимальная дистанция для взаимодействия с блоком

    def key_callback(self, window, key, scancode, action, mods):
        key_char = chr(key)
        if key_char in self.key_states:
            self.key_states[key_char] = (action == glfw.PRESS or action == glfw.REPEAT)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x, self.last_y = xpos, ypos

        sensitivity = 0.1
        self.yaw -= xoffset * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch + yoffset * sensitivity))
    
    def mouse_button_callback(self, window, button, action, mods):
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.break_block()
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.place_block()

    def update_movement(self):
        player_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'player')
        # ...existing code...
        # from scipy.spatial.transform import Rotation as R
        # quat = R.from_euler('zyx', [self.yaw, self.pitch, 0], degrees=True).as_quat()
        # self.data.body(player_body_id).xquat = [quat[3], quat[0], quat[1], quat[2]]

        # --- Движение по направлению взгляда ---
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        # Вектор вперед (с учетом yaw и pitch)
        forward = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        ])
        # Вектор вправо (только по горизонтали, без pitch)
        right = np.array([
            -math.sin(yaw_rad),
            math.cos(yaw_rad),
            0
        ])

        force = np.zeros(3)
        if self.key_states['W']: force += forward * self.move_speed
        elif self.key_states['S']: force -= forward * self.move_speed        

        if self.key_states['A']: force += right * self.move_speed
        elif self.key_states['D']: force -= right * self.move_speed

        self.data.xfrc_applied[player_body_id, :3] = force[:3]

        if self.key_states[' ']:
            self.data.xfrc_applied[player_body_id, 2] += self.jump_force
            self.key_states[' '] = False # Чтобы прыжок был одиночным

    def break_block(self):
        hit = self.raycast_block()
        if hit:
            x, z, y = hit
            if self.world.blocks[x, z, y] != BLOCK_TYPES["air"]:
                self.world.blocks[x, z, y] = BLOCK_TYPES["air"]
                self.world.update_model()
                self.model = self.world.model
                self.data = self.world.data
                self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
                self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def place_block(self):
        hit = self.raycast_block(return_prev=True)
        if hit:
            x, z, y = hit
            if self.world.blocks[x, z, y] == BLOCK_TYPES["air"]:
                self.world.blocks[x, z, y] = BLOCK_TYPES["dirt"]
                self.world.update_model()
                self.model = self.world.model
                self.data = self.world.data
                self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
                self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    def raycast_block(self, return_prev=False):
        # Получить позицию игрока и направление взгляда
        player_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'player')
        pos = self.data.body(player_body_id).xpos.copy()
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        dir_vec = np.array([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        ])
        step = 0.1
        prev = None
        for i in np.arange(0, self.break_distance, step):
            p = pos + dir_vec * i
            x, z, y = int(round(p[0])), int(round(p[1])), int(round(p[2]))
            if 0 <= x < self.world.size[0] and 0 <= z < self.world.size[1] and 0 <= y < self.world.size[2]:
                if self.world.blocks[x, z, y] != BLOCK_TYPES["air"]:
                    return (prev if return_prev else (x, z, y))
                prev = (x, z, y)
        return None

    def run(self):
        if not glfw.init():
            raise Exception("Could not initialize GLFW")
        self.window = glfw.create_window(1280, 720, "Voxel Game", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Could not create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        player_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'player')

        print_instructions()
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < 1.0/60.0:
                self.update_movement()
                mujoco.mj_step(self.model, self.data)
            
            # --- Вид от первого лица ---
            # Позиция камеры = позиция игрока + небольшой сдвиг вверх
            player_pos = self.data.body(player_body_id).xpos.copy()
            cam_height_offset = 0.0  # центр цилиндра
            cam_pos = player_pos + np.array([0, 0, cam_height_offset])
            self.cam.lookat = cam_pos

            # Направление взгляда по yaw/pitch
            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)

            self.cam.distance = 0.01  # минимальная дистанция, чтобы камера была "внутри" игрока
            self.cam.azimuth = self.yaw
            self.cam.elevation = self.pitch

            viewport = mujoco.MjrRect(0, 0, *glfw.get_framebuffer_size(self.window))
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)
            
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
        glfw.terminate()

def create_dummy_textures():
    if not os.path.exists("textures"): os.makedirs("textures")
    texture_files = {"grass.png": (34, 139, 34), "dirt.png": (139, 69, 19), "stone.png": (112, 128, 144), "bedrock.png": (80, 80, 80)}
    for filename, color in texture_files.items():
        if not os.path.exists(f"textures/{filename}"):
            Image.new('RGB', (64, 64), color=color).save(f"textures/{filename}")

def print_instructions():
    print("--- MuJoCo Voxel Game ---")
    print("Управление:")
    print("  W, A, S, D:      Перемещение")
    print("  Пробел:          Прыжок")
    print("  Мышь:            Осмотр")
    print("  ESC:             Выход")
    print("-------------------------")

if __name__ == '__main__':
    logging.info("Запуск основной программы.")
    create_dummy_textures()
    game = VoxelGame()
    game.run()
if __name__ == '__main__':
    logging.info("Запуск основной программы.")
    create_dummy_textures()
    game = VoxelGame()
    game.run()
