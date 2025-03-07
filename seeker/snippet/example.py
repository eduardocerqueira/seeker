#date: 2025-03-07T16:49:40Z
#url: https://api.github.com/gists/477c6510da0662a701005fdd8d4be2bd
#owner: https://api.github.com/users/logdog

import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2  # pip install opencv-python


# Input
class_list = [ 'pizza' , 'cat' , 'bus']
# ##########################
# Mapping from COCO label to Class indices
coco_labels_inverse = {}
ann_file = r'./data/COCO/annotations_trainval2014/annotations/instances_train2014.json'
coco = COCO(ann_file)

pizzaCatId = coco.getCatIds(catNms='pizza')[0]
pizzaImgIds = coco.getImgIds(catIds=pizzaCatId)
# print(f'{pizzaImgIds=}')

catCatId = coco.getCatIds(catNms='cat')[0]
catImgIds = coco.getImgIds(catIds=catCatId)
# print(f'{catImgIds=}')

busCatId = coco.getCatIds(catNms='bus')[0]
busImgIds = coco.getImgIds(catIds=busCatId)
# print(f'{busImgIds=}')

catIds = [pizzaCatId, catCatId, busCatId]
categories = coco.loadCats(ids=catIds)
categories.sort(key = lambda x :x[ 'id'])
print(f'{categories=}')

coco_labels_inverse = {}
for idx, in_class in enumerate(class_list):
    for c in categories :
        if c['name'] == in_class :
            coco_labels_inverse[c['id']]= idx
print(f'{coco_labels_inverse=}')

# ############################
# Retrieve Image list (for pizza)

imgIds = coco.getImgIds(catIds = [pizzaCatId, catCatId])

# ############################
# Display one random image with annotation
idx = np.random.randint(0, len(imgIds))
img = coco.loadImgs(imgIds[idx])[0 ]
I = io.imread(img['coco_url'])
# change from grayscale to color
if len(I.shape)==2:
    I = skimage.color.gray2rgb(I)

# pay attention to the flag, iscrowd being set to False
annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds,
iscrowd = False)
anns = coco.loadAnns(annIds)
fig, ax = plt.subplots(1,1)
image = np.uint8(I)
for ann in anns :
    [x,y,w,h] = ann['bbox']
    label =coco_labels_inverse[ ann['category_id'] ]
    image = cv2.rectangle(image,(int(x), int(y)),(int(x+w),
    int(y+h)),(36, 255, 12),2)
    image = cv2.putText(image, class_list[label],(int(x), int(
    y - 10)), cv2.
    FONT_HERSHEY_SIMPLEX,
    0.8,(36, 255, 12),2)
    
ax.imshow(image)
ax.set_axis_off()
plt.axis('tight')
plt.show()