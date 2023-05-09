#date: 2023-05-09T16:42:29Z
#url: https://api.github.com/gists/6a7ec76d9ba618c595a842c41a13b4da
#owner: https://api.github.com/users/lukad

# Here's a more advanced method that leverages the CIE 1931 XYZ color space
# and CIE Lab* color space to convert RGB to RGBW. This approach considers the color
# temperature of the white LED and takes into account human perception of color.

def gamma_correction(value):
    if value <= 0.04045:
        return value / 12.92
    else:
        return ((value + 0.055) / 1.055) ** 2.4

def inverse_gamma_correction(value):
    if value <= 0.0031308:
        return value * 12.92
    else:
        return 1.055 * (value ** (1 / 2.4)) - 0.055

def rgb_to_xyz(r, g, b):
    r = gamma_correction(r / 255)
    g = gamma_correction(g / 255)
    b = gamma_correction(b / 255)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x, y, z

def xyz_to_rgb(x, y, z):
    r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
    b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252

    r = inverse_gamma_correction(r) * 255
    g = inverse_gamma_correction(g) * 255
    b = inverse_gamma_correction(b) * 255

    return r, g, b

def find_best_white(rgb, white_points, tolerance=0.01):
    target_xyz = rgb_to_xyz(*rgb)
    target_y = target_xyz[1]
    best_white = None
    min_error = float('inf')

    for white in white_points:
        white_rgb = (white[0], white[1], white[2])
        white_xyz = rgb_to_xyz(*white_rgb)
        white_y = white_xyz[1]
        error = abs(target_y - white_y)

        if error < tolerance and error < min_error:
            best_white = white
            min_error = error

    return best_white

def rgb_to_rgbw_advanced_no_dependencies(r, g, b, white_points):
    rgb = (r, g, b)
    best_white = find_best_white(rgb, white_points)

    if best_white:
        r_w, g_w, b_w = best_white
        min_rgbw = min(r_w, g_w, b_w)
        w = min_rgbw
        return (r - r_w, g - g_w, b - b_w, w)
    else:
        return (r, g, b, 0)

white_points = [
    (255, 191, 127),  # Modify these values to match the color temperature of your white LED
    # Add more white points if needed
]

r, g, b = 180, 100, 200  # Example RGB color

rgbw = rgb_to_rgbw_advanced_no_dependencies(r, g, b, white_points)
print(rgbw)