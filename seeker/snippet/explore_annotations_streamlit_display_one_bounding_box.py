#date: 2022-01-11T17:12:31Z
#url: https://api.github.com/gists/beeb412a324b3001bf4470945b89c9c8
#owner: https://api.github.com/users/Wirg

import pandas as pd
import streamlit as st

from src.utils.display import load_and_annotate_image


def load_all_annotations() -> pd.DataFrame:
    return pd.read_parquet("data/annotations/coco_val_2020.parquet.gzip")


# Select the sad toaster as an example
# You can use .iloc[:1] on your own dataset instead
image_annotations = (
    load_all_annotations()
    .loc[lambda df: df.image_name == "000000066841.jpg"]
    .loc[lambda df: df.category_name == "toaster"]
)
coco_url = image_annotations["coco_url"].iloc[0]
image_name = image_annotations["image_name"].iloc[0]
st.image(
    load_and_annotate_image(
        coco_url,
        image_annotations,
        # make the bounding box blue
        color_map={"toaster": (0, 0, 255)}
    ),
    caption=image_name,
)
