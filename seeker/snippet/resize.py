#date: 2024-09-10T16:42:12Z
#url: https://api.github.com/gists/f8c671c358e45348b2e6fbbd523a78c5
#owner: https://api.github.com/users/samhaswon

import argparse
import multiprocessing
import os
import re
import time

from PIL import Image
from PIL.Image import Image as PILImage


def resize_image(img_path: str, max_size: int) -> None:
    """
    Resize a given image if it is greater than the specified size.
    :param img_path: The path to the image to resize.
    :param max_size: The maximum side length of the image.
    :return: None
    """
    img: PILImage = Image.open(img_path)
    if max(img.size) > max_size:
        scale_factor = max_size / max(img.size)
        img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)))
        img.save(img_path)
    img.close()


if __name__ == '__main__':
    start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="A script to convert all of the images in a directory to a given max side length."
    )

    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="The maximum side length of processed images"
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="The directory with images to process"
    )

    args = parser.parse_args()

    assert os.path.isdir(args.directory), f"Input directory `{args.directory}` is not a directory"

    print(f"Using path {args.directory} and size {args.size}")

    # Get the list of image files
    file_list = [
        x.path
        for x in os.scandir(args.directory)
        if os.path.isfile(x.path) and re.search(r"\.(png|jpe?g|bmp|webp|jfif)$", x.path, re.IGNORECASE)
    ]

    print(f"Found {len(file_list)} images")

    # Initialization of pool
    num_processes = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=num_processes)

    # Add files to the pool
    for file in file_list:
        pool.apply_async(resize_image, args=(file, args.size,))

    # Start, do the work, and wait for results
    pool.close()
    pool.join()
    end = time.perf_counter()
    print("Done")
    print(f"Took {end - start:.3f} seconds")
