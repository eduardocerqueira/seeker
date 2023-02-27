#date: 2023-02-27T17:04:32Z
#url: https://api.github.com/gists/fcceaf662084f9b2362891a5a9da1e97
#owner: https://api.github.com/users/jeffreyhooperjj

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches

def shoelace_formula(vertices):
    n = len(vertices)
    x = [vertices[i][0] for i in range(n)]
    y = [vertices[i][1] for i in range(n)]
    area = 0.5 * abs(sum([x[i] * y[(i + 1) % n] for i in range(n)]) - sum([y[i] * x[(i + 1) % n] for i in range(n)]))
    return area

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


center_cell = "TC"
target_cell = "TREG"

dots = False
counts = not dots

xgridsize = 14


x = all_cells_df["Centroid X µm"].values
y = all_cells_df["Centroid Y µm"].values

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

### logic to add in InRangeCount for T cells ###
fname = f"{path}/{core_name}_data.txt_&_center={center_cell}_&_target={target_cell}_&_pixel_dist=60.out.csv"
keep_cells_df = pd.read_csv(fname)
keep_cells_df = keep_cells_df.rename(columns={"RowId": "Row ID"})
columns = ["Row ID", "Name", "CentroidX", "CentroidY", "InRangeCount"]
keep_cells_df = keep_cells_df[columns]
mapping_dict = dict(zip(keep_cells_df["Row ID"], keep_cells_df["InRangeCount"]))
all_cells_df.index += 1
all_cells_df.index.name = "Row ID"
all_cells_df["InRangeCount"] = all_cells_df.index.map(mapping_dict)
# all_cells_df.replace(np.nan, -1, inplace=True)
print(keep_cells_df)


def reduce_function(arr):
    count = 0
    total = 0.0
    for num in arr:
        if num >= 0:
            count+=1
            total += num
    if count != 0:
        return float(total / count)
    return -1


colorbar_max_val = 25
new_xgridsize = xgridsize

### iterative process to try to equate sizes of hexagons ###
if counts:
    ratio_closest_to_1 = 10000
    ratio = -1
    while True:
        # if abs(1 - ratio) > abs(1 -ratio_closest_to_1):
        #     break
        fig2 = axs.hexbin(x, y, cmap=prev_cmap, vmin=0, vmax=colorbar_max_val, C=all_cells_df["InRangeCount"],
                  reduce_C_function=reduce_function, mincnt=0, gridsize=new_xgridsize, edgecolors="black")
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
                # try to make hexagons bigger
                new_xgridsize -= 1
            else:
                # try to make hexagons smaller
                new_xgridsize += 1
        else:
            # found closest match
            break
            
        print(f"Ratio: {ratio_closest_to_1=}")
    print(f"{id(fig2)=} {id(final_fig2)=}")
    fig2 = final_fig2
    fig2.set_visible(True)
    print(f"{id(fig2)=} {id(final_fig2)=}")
# fig2 = axs.hexbin(x, y, cmap=prev_cmap, vmin=0, vmax=colorbar_max_val, C=all_cells_df["InRangeCount"],
                  # reduce_C_function=reduce_function, mincnt=0, gridsize=14, edgecolors="black")

# Set the axis limits and labels
plt.xlim(min(xmin, min(x)), max(xmax, max(x)))
plt.ylim(min(ymin, min(y)), max(ymax, max(y)))

fig2_hexagon_values = fig2.get_array()
fig2_hexagon_locations = fig2.get_offsets()

print(min(fig2_hexagon_values))
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

print(hex_dict)

# print(fig1_hexagon_locations)
# print("********After****")
# print(fig2_hexagon_locations)
# print(f"MAX {max(fig2_hexagon_values)}")
for i, count in enumerate(fig2_hexagon_values):
    fig2_hexagon_values[i] = round(count, 1)

### adds count to middle of hexagon ###
if (counts):
    for i, xy in enumerate(fig2_hexagon_locations):
        # needed to remove hexagons that overlap
        if fig2_hexagon_values[i] < 0:
            fig2_hexagon_locations[i] = 0,0
        # else:
        #     # pass
        x = xy[0]
        y = xy[1]
        plt.text(x, y, fig2_hexagon_values[i], fontsize=5, horizontalalignment="center")


# hide hexbin plots
### uncomment 3 lines below to hide plots ###
# fig1.set_visible(False)
# fig2.set_visible(False)
# plt.draw()

### adds dots of each cell location ###
if (dots):
    groups = all_cells_df.groupby("Class")
    for name, group in groups:
        if name == "TC" or name == "TCp":
            axs.scatter(group["Centroid X µm"], group["Centroid Y µm"], s=3, label=name)
        # if name != "other":
        #     axs.scatter(group["Centroid X µm"], group["Centroid Y µm"], s=3, label=name)

xcenter = (xmin + xmax)/ 2
ycenter = (ymin + ymax)/2

radius = xcenter-xmin if (xcenter-xmin>ycenter-ymin) else ycenter-ymin



axs.add_patch(patches.Circle((xcenter,ycenter),
                            radius=radius,
                            color='k', linewidth=1, fill=False))
if (dots):
    plt.legend(markerscale=2, fontsize=7)
axs.set_title(f"Distribution of {core_name}")
fig.colorbar(fig2, label=f"Average {center_cell} vs {target_cell} Associations")
print("1 hexagons:",len(fig1_hexagon_locations))
print("2 hexagons:", len(fig2_hexagon_locations))
# fig.clim(0,50)

print(f"Chosen Ratio: {ratio_closest_to_1}")
# print(fig2.get_paths()[0])
hexagon_df = pd.DataFrame()
hexagon_df["hexagon_xy_centers"] = [f"{xy[0]} {xy[1]}" for xy in fig2_hexagon_locations]
hexagon_df[f"Average {center_cell} vs {target_cell} Associations"] = fig2_hexagon_values
hexagon_df.loc[hexagon_df[f"Average {center_cell} vs {target_cell} Associations"] < 0, f'f"Average {center_cell} vs {target_cell} Associations"'] = 0
hexagon_df["total white hex count"] = len(fig1_hexagon_locations)
hexagon_df["total colored hex count"] = len(fig2_hexagon_locations)
hexagon_df.to_excel(f"./excel/{core_name}_{center_cell}_vs_{target_cell}_association_hexagon_data.xlsx", index=False)
plt.savefig(f"./pix/{core_name}_data.txt_&_center={center_cell}_&_target={target_cell}_&_pixel_dist=60_&dots={dots}_&_counts={counts}_association.png", dpi=400)
plt.show()

