#date: 2025-07-21T17:08:37Z
#url: https://api.github.com/gists/b2e1664f2c88b7b8fe9968d50fa4a043
#owner: https://api.github.com/users/Cheaterman

def calibrate(points: list[tuple[float, float]]):
    global CALIBRATION_MATRIX
    from kivy.core.window import Window

    matrix = Matrix()
    bottom_left, bottom_right, top_right = points

    angle = math.atan2(
        bottom_right[1] - bottom_left[1],
        bottom_right[0] - bottom_left[0],
    )

    # Scale to screen size
    diagonal_vector = Vector(
        top_right[0] - bottom_left[0],
        # Y is inverted
        bottom_left[1] - top_right[1],
    ).rotate(math.degrees(angle))
    scale_vector = (
        Vector(Window.width, Window.height)
        / Vector(diagonal_vector)
    )
    matrix.scale(scale_vector[0], scale_vector[1], 0)

    # Rotate according to bottom points
    matrix.rotate(angle, 0, 0, 1)

    # Offset in the newly rotated/scaled space
    origin = Vector(
        -bottom_left[0] * scale_vector[0],
        # Y is inverted
        (-Window.height + bottom_left[1]) * scale_vector[1],
    ).rotate(math.degrees(angle))

    # Translate to new origin
    matrix.translate(origin[0], origin[1], 0)

    CALIBRATION_MATRIX = matrix
