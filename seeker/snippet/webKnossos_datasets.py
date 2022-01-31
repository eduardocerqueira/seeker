#date: 2022-01-31T17:02:20Z
#url: https://api.github.com/gists/1738c147e629286cf8a4f2089e6c69d2
#owner: https://api.github.com/users/hotzenklotz

import numpy as np

import webknossos as wk

# pylint: disable=unused-variable


def main() -> None:
    #####################
    # Opening a dataset #
    #####################

    dataset = wk.Dataset.open("testdata/simple_wk_dataset")
    # Assuming that the dataset has a layer "color"
    # and the layer has the magnification 1
    layer = dataset.get_layer("color")
    mag1 = layer.get_mag("1")

    ######################
    # Creating a dataset #
    ######################

    dataset = wk.Dataset("testoutput/my_new_dataset", scale=(1, 1, 1))
    layer = dataset.add_layer(
        layer_name="color", category="color", dtype_per_channel="uint8", num_channels=3
    )
    mag1 = layer.add_mag("1")
    mag2 = layer.add_mag("2")

    ##########################
    # Writing into a dataset #
    ##########################

    # The properties are updated automatically
    # when the written data exceeds the bounding box in the properties
    mag1.write(
        absolute_offset=(10, 20, 30),
        # assuming the layer has 3 channels:
        data=(np.random.rand(3, 512, 512, 32) * 255).astype(np.uint8),
    )

    mag2.write(
        absolute_offset=(10, 20, 30),
        data=(np.random.rand(3, 256, 256, 16) * 255).astype(np.uint8),
    )

    ##########################
    # Reading from a dataset #
    ##########################

    data_in_mag1 = mag1.read()  # the offset and size from the properties are used
    data_in_mag1_subset = mag1.read(absolute_offset=(10, 20, 30), size=(512, 512, 32))

    data_in_mag2 = mag2.read()
    data_in_mag2_subset = mag2.read(absolute_offset=(10, 20, 30), size=(512, 512, 32))
    assert data_in_mag2_subset.shape == (3, 256, 256, 16)

    #####################
    # Copying a dataset #
    #####################

    copy_of_dataset = dataset.copy_dataset(
        "testoutput/copy_of_dataset",
        block_len=8,
        file_len=8,
        compress=True,
    )
    new_layer = dataset.add_layer(
        layer_name="segmentation",
        category="segmentation",
        dtype_per_channel="uint8",
        largest_segment_id=0,
    )
    # Link a layer of the initial dataset to the copy:
    sym_layer = copy_of_dataset.add_symlink_layer(new_layer)


if __name__ == "__main__":
    main()