#date: 2021-11-26T17:09:09Z
#url: https://api.github.com/gists/19e180d0b4b7ecf5d195738bb57dec8c
#owner: https://api.github.com/users/mrtj

from yolox_postprocess import demo_postprocess, multiclass_nms

def process_results(self, inference_results, stream, ratio):
    media_height, media_width, _ = stream.image.shape
    media_scale = np.asarray([media_width, media_height, media_width, media_height])
    for output in inference_results:
        boxes, scores, class_indices = self.postprocess(
            output, self.MODEL_INPUT_SIZE, ratio)
        for box, score, class_idx in zip(boxes, scores, class_indices):
            if score * 100 > self.threshold:
                (left, top, right, bottom) = np.clip(box / media_scale, 0, 1)
                stream.add_rect(left, top, right, bottom)

def postprocess(self, result, input_shape, ratio):
    input_size = input_shape[-2:]
    predictions = demo_postprocess(result, input_size)
    predictions = predictions[0] # TODO: iterate through eventual batches
    
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = \
            dets[:, :4], dets[:, 4], dets[:, 5]
    
    boxes = final_boxes
    scores = final_scores
    class_indices = final_cls_inds.astype(int)
    return boxes, scores, class_indices