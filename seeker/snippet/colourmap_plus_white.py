#date: 2024-03-01T16:59:19Z
#url: https://api.github.com/gists/007e10138d859fd7c7d68d53c26961d8
#owner: https://api.github.com/users/graptolite

import matplotlib as mpl
import numpy as np

def coloumap_plus_white(colourmap_name,N,new_colourmap_name=""):
    # Local function to convert RGB colour format into hex colour format.
    # The value of each band must be an integer.
    rgb2hex = lambda r,g,b : "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
    # Define number of non-white colourmap levels.
    n_intermediates = N - 1
    # Find the RGB (with each band ranging from [0,255]) colours at the number of equally spaced intervals desired.
    intermediates = np.array([mpl.colormaps[colourmap_name](int(i)) for i in np.linspace(0,255,n_intermediates)])*255
    # Convert the RGB colours into hex colours.
    hex_colors = [rgb2hex(*c) for c in intermediates[:,:-1]]
    # Construct colourmap using this list of hex colours, with white as the starting colour.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(new_colourmap_name,["white"]+hex_colors,N=N)
    return cmap

# Usage Examples:
# Create colourmap similar to matplotlib's Reds but with the lowest level being white.
cmap1 = coloumap_plus_white("Reds",10,"whitePlusReds")