#date: 2021-09-01T17:17:33Z
#url: https://api.github.com/gists/7d4b29cc5414fa172138df1dff8ad478
#owner: https://api.github.com/users/sylvchev

from moabb.datasets import Cho2017, BNCI2014001, PhysionetMI
from moabb.paradigms import MotorImagery
from moabb.datasets.utils import find_intersecting_channels

datasets = [Cho2017(), BNCI2014001(), PhysionetMI()]

common_channels, _ = find_intersecting_channels(datasets)
chans = common_channels[:3]
paradigm = MotorImagery(channels=chans)

for d in datasets:
    ep, _, _ = paradigm.get_data(dataset=d, subjects=[1], return_epochs=True)
    print(f"{d.code} channel order: {ep.info['ch_names']}")
