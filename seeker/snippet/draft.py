#date: 2022-05-17T17:00:43Z
#url: https://api.github.com/gists/2ca4685f0ba33a0768dcf04cd3f247f6
#owner: https://api.github.com/users/saudzahirr

from manimce import *


# Colors.
CURVE_COLOR = BLUE

# Helpers.

def number_plane(x, y):
    plane = NumberPlane(
        x_range = [-x, x],
        y_range = [-y, y],
        x_length = 2*x,
        y_length = 2*y,
        background_line_style = {
            "stroke_color": GREY_B,
            "stroke_opacity": 0.5,
            "stroke_width": 1,
        },
        faded_line_ratio = 2,
    )
    plane.axes.set_stroke(opacity=0.5)
    plane.add_coordinates(font_size=18)
    plane.add(SurroundingRectangle(plane, WHITE, buff = 0.0, stroke_width = 2))
    return plane


# Scenes

class Test(Scene):
  def construct(self):
    plane = number_plane(3.5, 3.5)
    plane.set_height(6.0)
    plane.scale(1.25)

    quadratic_curve = plane.plot(
        lambda x: x**2 + 2*x - 2,
        x_range = [-3.5, +1.55],
        color = CURVE_COLOR, stroke_width = 3
    )
    
    self.play(
        Create(plane),
        run_time = 2,
        rate_func = smooth
    )
    self.play(
        Write(quadratic_curve),
        run_time = 3,
        rate_func = smooth
    )
    self.wait()