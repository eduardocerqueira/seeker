#date: 2024-09-10T16:41:04Z
#url: https://api.github.com/gists/03a3e74d6a0637229ae515d69928eb1d
#owner: https://api.github.com/users/thomasahle

from manim import *
import numpy as np

class DualRingHeatTransferAnimation(Scene):
    def construct(self):
        # Configuration
        num_blocks = 20
        num_around = 39
        block_size = 0.4
        outer_radius = 3.3
        inner_radius = 2.9
        step_time = 0.3
        pause_time = 0.1
        color_change_time = 0.4
        num_steps = 2 * num_around
        num_steps = 13
        step_angle = TAU / num_around

        # Create outer and inner rings of blocks
        outer_blocks = VGroup(*[Square(side_length=block_size, stroke_width=1) for _ in range(num_blocks)])
        inner_blocks = VGroup(*[Square(side_length=block_size, stroke_width=1) for _ in range(num_blocks)])

        # Position blocks in circles
        def position_blocks(blocks, radius, sign=1, start_angle=0):
            for i, block in enumerate(blocks):
                angle = i * step_angle * sign + start_angle
                block.move_to([radius, 0, 0])
                block.rotate(angle, about_point=ORIGIN)

        position_blocks(outer_blocks, outer_radius, -1)
        position_blocks(inner_blocks, inner_radius, +1, step_angle)

        # Set initial temperatures and colors
        outer_temps = [1.0] * num_blocks
        inner_temps = [0.0] * num_blocks

        def get_color(temp):
            return color_gradient((BLACK, RED), 101)[int(temp * 100)]

        def update_colors(blocks, temps):
            for block, temp in zip(blocks, temps):
                block.set_fill(color=get_color(temp), opacity=1)

        update_colors(outer_blocks, outer_temps)
        update_colors(inner_blocks, inner_temps)

        # Add blocks to the scene
        self.add(outer_blocks, inner_blocks)

        # Animate the movement and heat transfer
        for offset in range(num_steps):
            # Move rings
            move_animations = [
                outer_blocks.animate.rotate(step_angle/2, about_point=ORIGIN),
                inner_blocks.animate.rotate(-step_angle/2, about_point=ORIGIN),
            ]
            self.play(*move_animations, run_time=step_time)

            # Pause movement
            self.wait(pause_time)

            # Perform heat transfer and prepare color change animations
            color_animations = []
            for i in range(num_blocks):
                outer_index = i % num_around
                inner_index = (offset - i) % num_around
                if inner_index >= num_blocks or outer_index >= num_blocks:
                    continue

                avg_temp = (outer_temps[outer_index] + inner_temps[inner_index]) / 2
                outer_temps[outer_index] = avg_temp
                inner_temps[inner_index] = avg_temp

                color_animations += [
                    outer_blocks[outer_index].animate.set_fill(color=get_color(avg_temp)),
                    inner_blocks[inner_index].animate.set_fill(color=get_color(avg_temp)),
                ]


            # Animate all color changes concurrently
            if color_animations:
                self.play(*color_animations, run_time=color_change_time)

            # Pause movement
            self.wait(pause_time)

        # Final wait
        self.wait(1)
