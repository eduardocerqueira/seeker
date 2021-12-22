#date: 2021-12-22T17:04:12Z
#url: https://api.github.com/gists/8b5f20efbcfed4a4e45603c5c20cbaf2
#owner: https://api.github.com/users/alexppppp

def create_yolo_annotations(mask_comp, labels_comp):
    comp_w, comp_h = mask_comp.shape[1], mask_comp.shape[0]
    
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
    masks = mask_comp == obj_ids[:, None, None]

    annotations_yolo = []
    for i in range(len(labels_comp)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        xc = (xmin + xmax) / 2
        yc = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin

        annotations_yolo.append([labels_comp[i] - 1, round(xc/1920, 5), round(yc/1080, 5), round(w/1920, 5), round(h/1080, 5)])

    return annotations_yolo