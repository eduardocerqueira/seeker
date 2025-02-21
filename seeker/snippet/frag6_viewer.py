#date: 2025-02-21T16:46:05Z
#url: https://api.github.com/gists/fff403301dea7f3cfd520ef787bfae8d
#owner: https://api.github.com/users/SuperOptimizer

import numpy as np
import zarr
from typing import List, Tuple
import napari
import skimage


class SimpleRGBVolumeViewer:
    def __init__(self,
                 zarr_paths: List[str],
                 start_coords: Tuple[int, int, int] = (0, 0, 0),
                 size_coords: Tuple[int, int, int] = None):
        """
        Initialize viewer for RGB volume visualization.

        Args:
            zarr_paths: Paths to three zarr files [red, green, blue]
            start_coords: (z,y,x) starting coordinates for chunk extraction
            size_coords: (z,y,x) size of chunk to extract
        """
        if len(zarr_paths) != 3:
            raise ValueError("Need exactly three zarr paths")

        # Load zarr arrays and get volume shape
        arrays = [zarr.open(path, mode='r') for path in zarr_paths]

        # Check dtype of first array
        print(f"Input data type: {arrays[0].dtype}")

        self.shape = arrays[0].shape

        # Validate coordinates
        self.start = start_coords

        # If size not provided, use full volume
        if size_coords is None:
            self.size = self.shape
        else:
            self.size = size_coords

        # Load individual channels
        self.channels = self._load_channels(arrays)

        # Initialize viewer
        self.viewer = napari.Viewer(ndisplay=3)

        # Add individual channels with specific colors
        self._add_color_channels()

    def _load_channels(self, arrays):
        """Load channels individually for separate coloring."""
        channels = []
        for i, arr in enumerate(arrays):
            # Extract chunk
            chunk = arr[
                    self.start[0]:self.start[0] + self.size[0],
                    self.start[1]:self.start[1] + self.size[1],
                    self.start[2]:self.start[2] + self.size[2]
                    ]

            # Calculate percentiles for cutting
            p_low = np.percentile(chunk, 50)
            p_high = np.percentile(chunk, 98)

            print(f"Channel {i} percentiles: 2%={p_low}, 98%={p_high}")

            # Apply percentile cutting
            chunk = np.clip(chunk, p_low, p_high)

            min_val = np.min(chunk)
            max_val = np.max(chunk)
            equalized = skimage.exposure.equalize_hist((chunk - min_val) / (max_val - min_val),32)
            chunk = (equalized*255).astype(np.uint8)
            channels.append(chunk)

            # Verify data
            print(
                f"Channel {i} stats after conversion: min={np.min(chunk)}, max={np.max(chunk)}, mean={np.mean(chunk):.2f}")

        return channels

    def _add_color_channels(self):
        """Add each channel with appropriate coloring."""
        colors = ['red', 'green', 'blue']
        names = ['Red Channel', 'Green Channel', 'Blue Channel']

        self.layers = []

        for i, (channel, color, name) in enumerate(zip(self.channels, colors, names)):
            layer = self.viewer.add_image(
                channel,
                name=name,
                colormap=color,  # Apply specific color
                blending='additive',  # Allows colors to blend
                rendering='iso',  # Use isosurface rendering
                iso_threshold=64  # Initial threshold
            )
            self.layers.append(layer)

        print("Added 3 separate channels with individual coloring")

    def run(self):
        """Start the viewer."""
        print(f"Starting viewer with 3 colored channels")
        napari.run()

'''
rsync -av rsync://dl.ash2txt.org/data fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/53keV_3.24um_.zarr/3/ 53kev_3
rsync -av rsync://dl.ash2txt.org/data fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/70keV_3.24um_.zarr/3/ 70kev_3
rsync -av rsync://dl.ash2txt.org/data fragments/Frag6/PHerc51Cr4Fr8.volpkg/volumes_zarr/88keV_3.24um_.zarr/3/ 88kev_3
'''

if __name__ == "__main__":
    # Example usage
    paths = [
        "/Users/forrest/frag6/53kev_4",  # Red channel
        "/Users/forrest/frag6/70kev_4",  # Green channel
        "/Users/forrest/frag6/88kev_4"  # Blue channel
    ]

    viewer = SimpleRGBVolumeViewer(paths)
    viewer.run()