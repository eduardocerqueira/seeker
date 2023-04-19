#date: 2023-04-19T16:44:49Z
#url: https://api.github.com/gists/604ce5f946c213bebe011fd3276d85d4
#owner: https://api.github.com/users/jrkerns

# We assume there is a central picket in the image
# We use the demo image for repeatability even though it's not representative of your specific image
# REMEMBER: leaves are enumerated from the bottom up. I.e. leaf 12 is near the bottom of the image, not the top
# This might not correspond with reality, but that's the convention as it stands currently.

# setup
pf = pylinac.PicketFence.from_demo()
pf.analyze()

# method 1: Get picket distances from the CAX:
# this info is given in the results() call
print(pf.results())
# this can be obtained directly by doing:
offsets_from_cax = [picket.dist2cax for picket in pf.pickets]
# results for demo image: [59.53356528900607, 44.64335493252849, 29.595253246513945, 14.609400288020067, -0.465870473557089, -15.40415222753418, -30.427119859267613, -45.44139562896714, -60.41504350735243, -75.3628014370933]

# method 2: Get picket distances relative to a specific picket.
# this will create a dictionary with keys that are the picket number and the values are the distances to the central picket
picket_avg_positions = {}
# iterate through each picket
for picket_num in range(pf.num_pickets):
    mlc_measurements_at_picket = [meas.position[0] for meas in pf.mlc_meas if meas.picket_num == picket_num]
    avg_picket_position = np.mean(mlc_measurements_at_picket)
    # convert to physical distance
    picket_avg_positions[picket_num] = avg_picket_position / pf.image.dpmm
central_picket = picket_avg_positions[4]  # change this to be the index you want. For a 5-picket analysis it would be 2
# creat the final dict that has the positions relative to central picket
positions_relative_to_central = {picket_num: pos - central_picket for picket_num, pos in picket_avg_positions.items()}
print(positions_relative_to_central)
# for the demo image results are: {0: -59.99962818294546, 1: -45.10942577959054, 2: -30.06119730614614, 3: -15.075388023010248, 4: 0.0, 5: 14.938296101981877, 6: 29.961233730989562, 7: 44.97554203633322, 8: 59.94921235156443, 9: 74.89688725638564}

# method 3: Get distances between leaves by leaf pair
leaf_positions = {}
all_leaves = sorted(set(meas.full_leaf_nums[0] for meas in pf.mlc_meas))
# iterate through each leaf pair
central_leaf_idx = 4  # change to the index of interest. For a 5-picket image this would be 2
for leaf_num in all_leaves:
    sorted_leaves = sorted([meas.position[0] for meas in pf.mlc_meas if meas.leaf_num == leaf_num])  # we sort so we are always comparing each pair to the adjacent pair
    relative_leaf_positions = np.array(sorted_leaves) - sorted_leaves[central_leaf_idx]
    leaf_positions[leaf_num] = (relative_leaf_positions / pf.image.dpmm).tolist()  # convert to physical distance and convert back to simple list from numpy array.
print(leaf_positions)
# for the demo image results are: {12: [-60.047902783938646, -45.172806632261306, -30.10891160245866, -15.133696577454627, 0.0, 14.942249850151546, 29.937197205159855, 44.966515969703615, 59.95361613168376, 74.89103668181295], 13: [-60.00057341687474, -45.09665322768939, ...

# method 4: Get distances between each leaf by bank when running separate analysis
# this will compare all the A-leaf positions to the central A-leaf position
# only used when doing separate_leaves=True
pf.analyze(..., separate_leaves=True)

a_leaf_positions = {}
b_leaf_positions = {}
all_a_leaves = sorted(set(meas.full_leaf_nums[0] for meas in pf.mlc_meas))
all_b_leaves = sorted(set(meas.full_leaf_nums[1] for meas in pf.mlc_meas))
# iterate through each leaf
central_leaf_idx = 4  # change to the index of interest. For a 5-picket image this would be 2
# A leaves
for a_leaf_num in all_a_leaves:
    sorted_leaves = sorted([meas.position[0] for meas in pf.mlc_meas if meas.full_leaf_nums[0] == a_leaf_num])
    relative_leaf_positions = np.array(sorted_leaves) - sorted_leaves[central_leaf_idx]
    a_leaf_positions[a_leaf_num] = (relative_leaf_positions / pf.image.dpmm).tolist()
print(a_leaf_positions)

# B leaves
for b_leaf_num in all_b_leaves:
    sorted_leaves = sorted([meas.position[1] for meas in pf.mlc_meas if meas.full_leaf_nums[1] == b_leaf_num])
    relative_leaf_positions = np.array(sorted_leaves) - sorted_leaves[central_leaf_idx]
    b_leaf_positions[b_leaf_num] = (relative_leaf_positions / pf.image.dpmm).tolist()
print(b_leaf_positions)

# demo dataset gives: 
{'A12': [-59.92245356652932, -45.16703418970795, ...], 'A17': [-59.975295110257576, ...], ...}
{'B12': [-60.17335200134799, -45.1785790748147, ...], ...}



