#date: 2025-10-03T16:46:17Z
#url: https://api.github.com/gists/2bd2820e8d1c6fa64392d30c6e80afa1
#owner: https://api.github.com/users/bdpedigo

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from caveclient import CAVEclient

from nglui.statebuilder import ViewerState

client = CAVEclient("minnie65_phase3_v1")

cell_type_df = client.materialize.tables.cell_type_multifeature_v1().query(
    desired_resolution=[1, 1, 1],
    split_positions=True,
)

seg_version = 1412
cell_info = (
    client.materialize.views.aibs_cell_info()
    .query(
        materialization_version=seg_version,
        desired_resolution=[1, 1, 1],
        split_positions=True,
    )
    .set_index("id")
)
cell_info["cell_type_multifeature"] = cell_type_df.set_index("id")["cell_type"]
cell_info["broad_type_multifeature"] = cell_type_df.set_index("id")[
    "classification_system"
]

# %%
query_cell_info = cell_info.query("broad_type_multifeature == 'inhibitory'").copy()
print(len(query_cell_info), "inhibitory neurons")

#%%
currtime = time.time()

# passing in credentials was not necessary for me, but it may be for others...
# can look up polars storage_options for more details
storage_options = {
    "gcs.service_account_path": "**********"
}
synapses_from_cloud = (
    pl.scan_delta(
        "gs://allen-minnie-phase3/mat_deltalakes/v1412/synapses_pni_2_v1412_deltalake",
        # storage_options=storage_options,
    )
    .filter(
        pl.col("post_pt_root_id").is_in(query_cell_info["pt_root_id"].unique()),
        pl.col("post_pt_root_id") != pl.col("pre_pt_root_id"),
    )
    .select(["post_pt_root_id", "pre_pt_root_id", "size"])
    .collect()
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
# takes ~128 seconds on my machine plugged into allen institute wifi
gged into allen institute wifi
