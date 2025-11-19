#date: 2025-11-19T17:02:56Z
#url: https://api.github.com/gists/2a22144fd287a2997b65a65971e8fbe9
#owner: https://api.github.com/users/broccoder

import numpy as np
from PIL import Image
import time


def _inverse_gray_code(x, dims):
    """
    Applies inverse Gray coding to an array of integers.
    This converts Gray-coded values back to standard binary.
    """

    mask = x >> 1
    for _ in range(dims):
        x ^= mask
        mask >>= 1
    return x


def decode(hilbert_indices: np.ndarray, num_bits: int, num_dims: int):
    """
    Decodes Hilbert integers into N-dimensional coordinates.

    Args:
        hilbert_indices (np.ndarray): 1D array of Hilbert integers.
        num_bits (int): Bits per dimension (order of the curve).
        num_dims (int): Number of dimensions (e.g., 2 for image, 3 for color).

    Returns:
        np.ndarray: Array of shape (N, num_dims) containing coordinates.
    """

    h = np.atleast_1d(hilbert_indices)
    n_points = h.shape[0]

    coords = [np.zeros(n_points, dtype=h.dtype) for _ in range(num_dims)]

    for i in range(num_bits):
        chunk_shift = i * num_dims
        chunk = (h >> chunk_shift) & ((1 << num_dims) - 1)
        chunk = _inverse_gray_code(chunk, num_dims)

        for d in range(num_dims):
            bit = (chunk >> d) & 1
            coords[d] |= bit << i

    for b in range(num_bits - 1, -1, -1):
        lower_bits_mask = (1 << b) - 1

        for d in range(num_dims - 1, -1, -1):
            mask = (coords[d] >> b) & 1

            coords[0] ^= mask * lower_bits_mask
            swap_mask = (1 - mask) * lower_bits_mask

            t = (coords[0] ^ coords[d]) & swap_mask
            coords[0] ^= t
            coords[d] ^= t

    return np.stack(coords, axis=1)


def main():
    print("Starting...")
    start_time = time.time()

    img_bits = 12
    img_dims = 2

    color_bits = 8
    color_dims = 3

    total_points = 2 ** (img_bits * img_dims)
    side_length = 2**img_bits

    print(f"Generating {total_points:,} pixels.")

    hilberts = np.arange(total_points, dtype=np.int64)

    print("Calculating 2D Image Coordinates...")
    locs_2d = decode(hilberts, img_bits, img_dims)

    print("Calculating 3D Color Values...")
    locs_3d = decode(hilberts, color_bits, color_dims)

    print("Mapping pixels to image array...")

    img_arr = np.zeros((side_length, side_length, 3), dtype=np.uint8)

    img_arr[locs_2d[:, 1], locs_2d[:, 0]] = locs_3d.astype(np.uint8)

    print("Saving to 'rgb_spectrum_hilbert.png'...")
    im = Image.fromarray(img_arr)
    im.save("rgb_spectrum_hilbert.png")

    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
