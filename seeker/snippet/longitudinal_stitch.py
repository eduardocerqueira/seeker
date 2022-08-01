#date: 2022-08-01T16:54:53Z
#url: https://api.github.com/gists/707ccb1100af14400fffcdf3ee347eba
#owner: https://api.github.com/users/d-saikrishna

import natsort
import glob

extension = 'png'
vert_imgs = glob.glob('vert/*.{}'.format(extension))
vert_imgs = natsort.natsorted(vert_imgs,reverse=False)

#natsort helps in listing the images the correct numerical order 1,2,3 and not 1,10,2,20
merged_image = vert_imgs[0]

for image in vert_imgs[1:]:
    print(image)
    merged_image = merge_images(merged_image, image,horizontal=True)

#Save the final image of the given day.
merged_image.save(date_string+'.png')