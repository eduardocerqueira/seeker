#date: 2023-02-27T17:01:28Z
#url: https://api.github.com/gists/347fa199dcf187e60741ff5c4038b9c8
#owner: https://api.github.com/users/jeffreyhooperjj

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches

def get_x_and_white_range(x, y):
    xmin = min(x)
    ymin = min(y)
    xmax = max(x)
    ymax = max(y)
    return [xmin, xmax, ymin, ymax]

def calculate_ratios(coordinates):
    xdistance = coordinates[1] - coordinates[0]
    ydistance = coordinates[3] - coordinates[2]
    xratio = xdistance / xgridsize
    # yratio = ydistance / ygridsize
    return xratio

# approximate area of 1 hexagon
def shoelace_formula(vertices):
    n = len(vertices)
    x = [vertices[i][0] for i in range(n)]
    y = [vertices[i][1] for i in range(n)]
    area = 0.5 * abs(sum([x[i] * y[(i + 1) % n] for i in range(n)]) - sum([y[i] * x[(i + 1) % n] for i in range(n)]))
    return area

xgridsize = 14

core_name = "I8"
path = "./TMA 36 Data/TMA36_visualizations/data"
all_cells_df = pd.read_csv(f"{path}/{core_name}_data.txt", sep="\t")

map_dict = {"TC": "TC", "TCp": "TC", 
            "BC": "BC", "BCp": "BC",
            "CK": "CK", "CKp": "CK",
            "CKpHLADR": "CK", "CKHLADR": "CK",
            "other": "other", "TREG": "TREG",
            "BREG": "BREG"}

all_cells_df['UniqueClass'] = all_cells_df['Class'].map(map_dict)

keep_cell = "TC"
keep_cells = ["TC", "TCp"]

dots_only = False
dots = dots_only
counts = not dots

keep_cells_df = all_cells_df[all_cells_df["UniqueClass"].isin([keep_cell])].reset_index()

x = all_cells_df["Centroid X µm"].values
y = all_cells_df["Centroid Y µm"].values

coordinates = get_x_and_white_range(x,y)
xratio = calculate_ratios(coordinates)
print("coordinates",coordinates)
print("xratio", xratio)

fig, axs = plt.subplots(1)
# # cm = "RdBu_r"
cdict = {
    'red': [(0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0)],
    'green': [(0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0)],
    'blue': [(0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0)],
}
cmap = matplotlib.colors.LinearSegmentedColormap("test", segmentdata=cdict)
# axs.hexbin(x, y, mincnt= 1, cmap=cmap, gridsize=14, linewidths=1)
# NOTE: C would be InRangeCount
prev_cmap = "jet"
fig1 = axs.hexbin(x, y, cmap=cmap,  mincnt=1, gridsize=xgridsize, edgecolors="black")
fig1_hexagon_locations = fig1.get_offsets()
fig1_verts = fig1.get_paths()[0].vertices
fig1_area = shoelace_formula(fig1_verts)
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

print(f"{xmin=}, {xmax=}, {ymin=}, {ymax=}")

center_cell = "TC"
target_cell = "TREG"


x = keep_cells_df["Centroid X µm"].values
y = keep_cells_df["Centroid Y µm"].values

coordinates = get_x_and_white_range(x,y)

def get_new_gridsize(coordinates):
    xdistance = coordinates[1] - coordinates[0]
    ydistance = coordinates[3] - coordinates[2]
    new_xgridsize = xdistance / xratio
    print(new_xgridsize)
    # new_ygridsize = ydistance / yratio_val
    return int(new_xgridsize)
print("x,y", x,y)

# new_xgridsize = get_new_gridsize(coordinates)
new_xgridsize = xgridsize

colorbar_max_val = 25

### iterative process to try to equate sizes of hexagons ###
if counts:
    ratio_closest_to_1 = 10000
    ratio = -1
    while True:
        # if abs(1 - ratio) > abs(1 -ratio_closest_to_1):
        #     break
        fig2 = axs.hexbin(x, y, cmap=prev_cmap, vmin=0, vmax=colorbar_max_val,
                          mincnt=1, gridsize=new_xgridsize, edgecolors="black")
        fig2.set_visible(False)
        fig2_verts = fig2.get_paths()[0].vertices
        fig2_area = shoelace_formula(fig2_verts)
        ratio = fig2_area/fig1_area
        print(ratio)
        print(f"{abs(1 - ratio)} {abs(1 -ratio_closest_to_1)}")

        if abs(1 - ratio) < abs(1 -ratio_closest_to_1):
            ratio_closest_to_1 = ratio
            final_fig2 = fig2
            
            # update new_gridsize
            if ratio < 1:
                new_xgridsize -= 1
            else:
                new_xgridsize += 1
        else:
            # found closest match
            break
        # if round(ratio, 1) == 1.0:
        #     break
        # else:
            
        print(f"Ratio: {ratio_closest_to_1=}")
    print(f"{id(fig2)=} {id(final_fig2)=}")
    fig2 = final_fig2
    fig2.set_visible(True)
    print(f"{id(fig2)=} {id(final_fig2)=}")


# Set the axis limits and labels
plt.xlim(min(xmin, min(x)), max(xmax, max(x)))
plt.ylim(min(ymin, min(y)), max(ymax, max(y)))

if counts:
    fig2_hexagon_values = fig2.get_array()
    fig2_hexagon_locations = fig2.get_offsets()

# print("********BEFORE*****")
# print(fig2_hexagon_locations)

def find_closest_hexagon(fig2_hexagon_location, fig1_hexagon_locations):
    fig2_hex_x = fig2_hexagon_location[0]
    fig2_hex_y = fig2_hexagon_location[1]
    closest_hexagon = fig2_hex_x, fig2_hex_y
    shortest_dist = float('inf')
    for hexagon in fig1_hexagon_locations:
        dist = ((hexagon[0] - fig2_hex_x) ** 2 + (hexagon[1] - fig2_hex_y) ** 2) ** 0.5
        # print(dist)
        if (dist < shortest_dist):
            shortest_dist = dist
            closest_hexagon = hexagon[0], hexagon[1]
    return closest_hexagon

hex_dict = {}
if counts:
    for i, hexagon in enumerate(fig2_hexagon_locations):
        closest_hexagon = find_closest_hexagon(hexagon, fig1_hexagon_locations)
        # check to see if there is a hexagon already at that location
        if hex_dict.get(closest_hexagon) != None:
            # add hexagon values together
            old_hex = hex_dict[closest_hexagon]
            print(f"Overlapping hexagons {i}: {old_hex}")
            fig2_hexagon_values[i] += fig2_hexagon_values[old_hex]
            fig2_hexagon_values[old_hex] = -1
        hex_dict[closest_hexagon] = i
        fig2_hexagon_locations[i] = closest_hexagon


    # print(fig1_hexagon_locations)
    # print("********After****")
    # print(fig2_hexagon_locations)
    # print(f"MAX {max(fig2_hexagon_values)}")
    for i, count in enumerate(fig2_hexagon_values):
        fig2_hexagon_values[i] = round(count, 1)

### adds count to middle of hexagon ###
if counts:
    for i, xy in enumerate(fig2_hexagon_locations):
        # needed to remove hexagons that overlap
        x = xy[0]
        y = xy[1]
        if fig2_hexagon_values[i] < 0:
            fig2_hexagon_locations[i] = -x,-y
            continue
        # else:
        #     # pass
        plt.text(x, y, fig2_hexagon_values[i], fontsize=5, horizontalalignment="center")


# hide hexbin plots
### uncomment 3 lines below to hide plots ###
if dots_only:
    fig1.set_visible(False)
#     fig2.set_visible(False)
#     plt.draw()

### adds dots of each cell location ###
if (dots):
    groups = all_cells_df.groupby("Class")
    for name, group in groups:
        if name in keep_cells:
            axs.scatter(group["Centroid X µm"], group["Centroid Y µm"], s=3, label=name)
        # if name != "other":
        #     axs.scatter(group["Centroid X µm"], group["Centroid Y µm"], s=3, label=name)

xcenter = (xmin + xmax)/ 2
ycenter = (ymin + ymax)/2

radius = xcenter-xmin if (xcenter-xmin>ycenter-ymin) else ycenter-ymin



axs.add_patch(patches.Circle((xcenter,ycenter),
                            radius=radius,
                            color='k', linewidth=1, fill=False))
if dots:
    plt.legend(markerscale=2, fontsize=7)
axs.set_title(f"Distribution of {core_name}")
if counts:
    fig.colorbar(fig2, label=f"{keep_cell} Density")
    print("1 hexagons:",len(fig1_hexagon_locations))
    print("2 hexagons:", len(fig2_hexagon_locations))
# fig.clim(0,50)

# axs.set_aspect("equal")


### attempts at trying to get hexagon area
# # Get the hexagon paths and calculate their areas
# hex_paths = fig1.get_paths()
# hex_width = fig1.get_offsets()[1,0] - fig1.get_offsets()[0,0]
# hex_side = hex_width * 2 / 3
# hex_areas = [(3 * np.sqrt(3) / 2) * hex_side**2 for _ in hex_paths]
# print(hex_areas)

# # Get the hexagon paths and calculate their areas
# hex_paths = fig2.get_paths()
# hex_width = fig2.get_offsets()[1,0] - fig2.get_offsets()[0,0]
# hex_side = hex_width * 2 / 3
# hex_areas = [(3 * np.sqrt(3) / 2) * hex_side**2 for _ in hex_paths]
# print(hex_areas)

# # get the x and y offsets of each hexagon
# hex_offsets = fig2.get_offsets()

# # get the widths of each hexagon
# hex_widths = fig2.get_widths()

# # calculate the area of each hexagon
# hex_areas = 3 * (np.sqrt(3) / 2) * (hex_widths ** 2)
# print(hex_areas)
###       End of attempt               ######

if counts:
    print(f"Chosen Ratio: {ratio_closest_to_1}")
    hexagon_df = pd.DataFrame()
    hexagon_df["hexagon_xy_centers"] = [f"{xy[0]} {xy[1]}" for xy in fig2_hexagon_locations]
    hexagon_df[f"{keep_cells} density"] = fig2_hexagon_values
    hexagon_df.loc[hexagon_df[f"{keep_cells} density"] < 0, f'{keep_cells} density'] = 0
    hexagon_df["total white hex count"] = len(fig1_hexagon_locations)
    hexagon_df["total colored hex count"] = len(fig2_hexagon_locations)
    # hexagon_df = hexagon_df[hexagon_df["average_TC_association"] >= 0]
    hexagon_df.to_excel(f"./excel/{core_name}_{keep_cells}_density_hexagon_data.xlsx", index=False)
plt.savefig(f"./pix/{core_name}_{keep_cells}_&_dotsonly={dots_only}_&_dots={dots}_&_counts={counts}_density_data.png", dpi=400)
plt.show()
