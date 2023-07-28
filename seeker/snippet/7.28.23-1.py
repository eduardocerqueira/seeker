#date: 2023-07-28T17:09:04Z
#url: https://api.github.com/gists/ca8b08eaea3692f5d0d1bee6569533dd
#owner: https://api.github.com/users/jimmyguerrero

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
dataset = foz.load_zoo_dataset("quickstart")
dataset.persistent = True
# Create a view
cats_view = (
    dataset
    .select_fields("ground_truth")
    .filter_labels("ground_truth", F("label") == "cat")
    .sort_by(F("ground_truth.detections").length(), reverse=True)
)
# Save the view
dataset.save_view("cats-view", cats_view)