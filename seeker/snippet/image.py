#date: 2022-06-02T17:24:45Z
#url: https://api.github.com/gists/9430176c036da313bd4e213c51b9173f
#owner: https://api.github.com/users/adityaa-codes


from PIL import Image, ImageFilter


def crop_image(pic):
    width, height = pic.size
    if width == height:
        return pic
    offset = int(abs(height - width) / 2)
    if width > height:
        pic = pic.crop([offset, 0, width - offset, height])
    else:
        pic = pic.crop([0, offset, width, height - offset])

    pic = pic.resize((1080, 1080), Image.LANCZOS)
    return pic


def resize_image(pic):
    width, height = pic.size
    if width == height:
        return pic
    if width > height:
        base_width = 1080
        width_percent = (base_width / float(pic.size[0]))
        hsize = int((float(pic.size[1]) * float(width_percent)))
        pic = pic.resize((base_width, hsize), Image.LANCZOS)
    else:
        base_height = 1080
        height_percent = (base_height / float(pic.size[1]))
        width_size = int((float(pic.size[0]) * float(height_percent)))
        pic = pic.resize((width_size, base_height), Image.LANCZOS)
    return pic


def blurr_image(pic):
    im1 = pic.filter(ImageFilter.GaussianBlur(radius=200))
    return im1


def overlay_image(path):
    image = Image.open(path)
    resized_image = resize_image(image)
    cropped_image = crop_image(image)
    blurred_image = blurr_image(cropped_image)
    width, height = resized_image.size
    offset = int(abs(height - width) / 2)
    if width > height:
        blurred_image.paste(resized_image, (0, offset))
    else:
        blurred_image.paste(resized_image, (offset, 0))
    blurred_image.save("final.jpg")


overlay_image(r"test.jpg")

