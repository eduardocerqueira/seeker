#date: 2023-01-30T17:00:36Z
#url: https://api.github.com/gists/b81e712f3a33c016042b92598f0d062c
#owner: https://api.github.com/users/beantowel

from PIL import Image


def resize_and_crop(img_path, modified_path, size, crop_type="top"):
    """
    Resize and crop an image to fit the specified size.

    args:
    img_path: path for the image to resize.
    modified_path: path to store the modified image.
    size: `(width, height)` tuple.
    crop_type: can be 'top', 'middle' or 'bottom', depending on this
    value, the image will cropped getting the 'top/left', 'middle' or
    'bottom/right' of the image to fit the size.
    raises:
    Exception: if can not open the file in img_path of there is problems
    to save the image.
    ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize(
            (size[0], int(round(size[0] * img.size[1] / img.size[0]))), Image.LANCZOS
        )
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, img.size[0], size[1])
        elif crop_type == "middle":
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)),
            )
        elif crop_type == "bottom":
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize(
            (int(round(size[1] * img.size[0] / img.size[1])), size[1]), Image.LANCZOS
        )
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, size[0], img.size[1])
        elif crop_type == "middle":
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1],
            )
        elif crop_type == "bottom":
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]), Image.LANCZOS)
    # If the scale is the same, we do not need to crop
    img.save(modified_path)


sizes = (
    (1200, 630),
    (1950, 1300),
    (420, 280),
    (160, 160),
)

for size in sizes:
    print(f"size={size}")
    resize_and_crop("./screenshot.png", f"./screenshot{size[0]}.png", size, "middle")
