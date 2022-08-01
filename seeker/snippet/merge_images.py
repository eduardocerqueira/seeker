#date: 2022-08-01T16:52:41Z
#url: https://api.github.com/gists/b4c93d7b094d18da82923c8b3a0d4298
#owner: https://api.github.com/users/d-saikrishna

def merge_images(file1, file2,horizontal):
    """Merge two images into one

    Input parameters
    file1: path to first image file
    file2: path to second image file
    horizontal: True if the images are to be stitched horizontally
    
    returns the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    if horizontal==True:
        result_width = width1 + width2
        result_height = max(height1, height2)
        result = Image.new('L', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
    else:
        result_height = height1 + height2
        result_width = max(width1,width2)
        result = Image.new('L', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(0, height1))

    #result.save('result.image')

    return result