#date: 2026-01-23T17:17:56Z
#url: https://api.github.com/gists/eedbbe1c6477d73be735fc3dc629ac66
#owner: https://api.github.com/users/DortyTheGreat

# paste into .minecraft\xaero\minimap\{WORLD}\dim%0

base_coords = (64, 64)
x_incr = 128
y_incr = 128

# 18x18 maps
x_pos_radius = 8
x_neg_radius = 9
y_pos_radius = 8
y_neg_radius = 9

Y_CONST = 100

for x in range(-x_neg_radius, x_pos_radius + 1):
    for y in range(-y_neg_radius, y_pos_radius + 1):
        X = base_coords[0] + x * x_incr
        Z = base_coords[1] + y * y_incr
        name = f"mapart[{x};{y}]"

        print(
            f"waypoint:{name}:K:{X}:{Y_CONST}:{Z}:4:false:0:gui.xaero_default:false:0:0:false"
        )
