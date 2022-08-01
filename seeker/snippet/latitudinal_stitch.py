#date: 2022-08-01T16:54:04Z
#url: https://api.github.com/gists/fe25438cbcc1c62f727fd428dffa0801
#owner: https://api.github.com/users/d-saikrishna

import glob
extension = 'image'

result = glob.glob('Tiles/*.{}'.format(extension))
# The images were named in such a way that they are in batches of vertical stretch.95+95+95

#There are 95 images per a vertical stretch
c = 0
count=1
no_images_vertically = 95
while c+no_images_vertically<=len(result):
    print(count)
    vert_images = result[c:c+no_images_vertically]
    vert_images.reverse()
    print(len(vert_images))
    merged_image = vert_images[0]
    for image in vert_images[1:]:
        merged_image = merge_images(merged_image, image,horizontal=False)
    c=c+no_images_vertically
    
    merged_image.save(r'vert/'+str(count)+'.png')
    count=count+1