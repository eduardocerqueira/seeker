#date: 2023-03-01T16:53:34Z
#url: https://api.github.com/gists/233c4d2fa09f0c6f6b92a8ebc197b9a4
#owner: https://api.github.com/users/jrkerns

import pylinac


pf = pylinac.PicketFence.from_demo_image()
pf.analyze(separate_leaves=True)
pf.plot_analyzed_image()

# plot a specific leaf pair
pf.plot_leaf_profile('A13', 2)

# get a specific leaf
leaf_meas = [m for m in pf.mlc_meas if 'A13' in m.full_leaf_nums]
print("First picket measurement error", leaf_meas[0].error)
leaf_meas[0].profile.plot()