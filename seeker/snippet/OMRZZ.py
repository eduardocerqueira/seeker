#date: 2025-04-01T17:10:17Z
#url: https://api.github.com/gists/c63d66b9b937c137b57fc1df070a9c8f
#owner: https://api.github.com/users/RHindges

from math import cos, sin, pi
import os
import sys
import warnings
import numpy as np
from panda3d.core import *
from shared_modules_across_frameworks.modules.panda3d_scene import Panda3D_Scene

sine_grating_motion_shader = [
    """#version 460
        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;

        out vec2 texcoord;

        void main() {
           gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
           texcoord = p3d_MultiTexCoord0;
        }
    """,

    """#version 460
        uniform float contrast;
        uniform float wavelength;
        uniform float x;
        uniform float y;
        uniform float pattern_orientation;
        uniform float offset;
        uniform sampler2D p3d_Texture0;
        in vec2 texcoord;
        uniform vec4 p3d_ColorScale;

        void main() {

            float x_ = (2*texcoord.x-x-1)*cos(-pattern_orientation*3.1415/180) - (2*texcoord.y-y-1)*sin(-pattern_orientation*3.1415/180);

            float y_ = (2*texcoord.x-x-1)*sin(-pattern_orientation*3.1415/180) + (2*texcoord.y-y-1)*cos(-pattern_orientation*3.1415/180);

            float r = sqrt(pow(2*texcoord.x-1, 2) + pow(2*texcoord.y-1, 2));
            float c;

            if (r > 1.0) c = 0.0;
            else {
                c = 0.5*(sin((x_ - offset)*2*3.1415/wavelength)*contrast+1.0);
                }

            if (c > 0.5) {c = 1.0;} else {c = 0.0;}


            vec4 color = texture(p3d_Texture0, texcoord);
            gl_FragColor = vec4(c*p3d_ColorScale[0], c*p3d_ColorScale[1], c*p3d_ColorScale[2], 1.0*p3d_ColorScale[3]);
        }
    """
]


class MyApp(Panda3D_Scene):
    def __init__(self, stimulus_module):
        angles = [90]
        stripe_widths = [0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4]
        stripe_contrasts = [1]
        directions = ['left', 'right']
        self.stimulus_index_names = []
        self.stimuli = []

        for angle in angles:
            for direction in directions:
                for stripe_contrast in stripe_contrasts:
                    for stripe_width in stripe_widths:
                        self.stimulus_index_names.append(f'{stripe_width}_{stripe_contrast}_{direction}_{angle}')
                        self.stimuli.append([stripe_width, stripe_contrast, direction, angle])

        Panda3D_Scene.__init__(self, stimulus_module)
        self.resting_time = 5  # 5s static grating
        self.motion_time = 20  # 15s moving grating
        self.repetitions = 5  # Five repetitions
        self.total_stimulus_time = self.repetitions * (self.motion_time * 2 + self.resting_time * 2)
        self.show_fish_circles = False

        self.stimulus_indices_ordered_package = range(
            len(self.stimulus_index_names))  # Used for package-based stimulus delivery
        self.stimulus_durations = [self.total_stimulus_time] * len(self.stimulus_index_names)

        # Compile the motion shader
        self.compiled_sine_grating_motion_shader = Shader.make(Shader.SLGLSL,
                                                               sine_grating_motion_shader[0],
                                                               sine_grating_motion_shader[1])

        self.make_floor_cards()
        for arena_index in range(self.shared["setup_config_dict"]["number_of_arenas"]):
            self.scene_dict[f"arena_floor_card{arena_index}"].setShader(self.compiled_sine_grating_motion_shader)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("contrast", 1)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("wavelength", 0.2)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("x", 0.0)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("y", 0.0)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("pattern_orientation", 0.0)
            self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("offset", 0.0)

        self.pattern_offset = [0 for _ in range(self.shared["setup_config_dict"]["number_of_arenas"])]

    def start_stimulus(self, arena_index, stimulus_index):
        self.pattern_offset[arena_index] = 0

        stripe_width, stripe_contrast, direction, angle = self.stimuli[stimulus_index]
        if arena_index == 4:
            print(f'Fish 4 sees: {stripe_width}, {stripe_contrast}, {direction}, {angle}')

    def stop_stimulus(self, arena_index, stimulus_index, stimulus_time, dt):
        pass


    def update_stimulus(self, arena_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_durations[stimulus_index]:
            self.shared[f"stimulus_module.next_stimulus_requested_arena{arena_index}"].value = 1
            return


        stripe_width, stripe_contrast, direction, angle = self.stimuli[stimulus_index]
        wavelength = stripe_width
        cycle_time = (self.motion_time * 2 + self.resting_time * 2)
        current_phase = stimulus_time % cycle_time

        if current_phase < self.motion_time:
            self.apply_left_stimulus(arena_index, stimulus_time, dt, stripe_width)  # Left movement
        elif current_phase < self.motion_time + self.resting_time:
            self.apply_rest_stimulus(arena_index, stimulus_time, dt, stripe_width)  # Static grating
        elif current_phase < 2 * self.motion_time + self.resting_time:
            self.apply_right_stimulus(arena_index, stimulus_time, dt, stripe_width)  # Right movement
        else:
            self.apply_rest_stimulus(arena_index, stimulus_time, dt, stripe_width)  # Static grating

    def apply_rest_stimulus(self, arena_index, stimulus_time, dt, stripe_width):
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("wavelength", stripe_width)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("contrast", 1)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("x", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("y", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("no_stim", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("offset", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("pattern_orientation", 0)

    def apply_left_stimulus(self, arena_index, stimulus_time, dt, stripe_width):
        speed = 0.1
        self.pattern_offset[arena_index] += speed * dt

        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("wavelength", stripe_width)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("contrast", 1)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("x", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("y", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("no_stim", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("offset", self.pattern_offset[arena_index])
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("pattern_orientation", 0)

    def apply_right_stimulus(self, arena_index, stimulus_time, dt, stripe_width):
        speed = 0.1
        self.pattern_offset[arena_index] += speed * dt

        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("wavelength", stripe_width)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("contrast", 1)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("x", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("y", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("no_stim", 0)
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("offset", self.pattern_offset[arena_index])
        self.scene_dict[f"arena_floor_card{arena_index}"].setShaderInput("pattern_orientation", 180)