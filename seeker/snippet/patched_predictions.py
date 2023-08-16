#date: 2023-08-16T17:07:09Z
#url: https://api.github.com/gists/a08c1cc1407699c27ba94bf5c7df9598
#owner: https://api.github.com/users/Y-T-G

def _predict_by_feat_single(self,
                            cls_score_list: List[Tensor],
                            bbox_pred_list: List[Tensor],
                            score_factor_list: List[Tensor],
                            mlvl_priors: List[Tensor],
                            img_meta: dict,
                            cfg: ConfigDict,
                            rescale: bool = False,
                            with_nms: bool = True) -> InstanceData:
    """Transform a single image's features extracted from the head into
    bbox results.

    Args:
        cls_score_list (list[Tensor]): Box scores from all scale
            levels of a single image, each item has shape
            (num_priors * num_classes, H, W).
        bbox_pred_list (list[Tensor]): Box energies / deltas from
            all scale levels of a single image, each item has shape
            (num_priors * 4, H, W).
        score_factor_list (list[Tensor]): Score factor from all scale
            levels of a single image, each item has shape
            (num_priors * 1, H, W).
        mlvl_priors (list[Tensor]): Each element in the list is
            the priors of a single level in feature pyramid. In all
            anchor-based methods, it has shape (num_priors, 4). In
            all anchor-free methods, it has shape (num_priors, 2)
            when `with_stride=True`, otherwise it still has shape
            (num_priors, 4).
        img_meta (dict): Image meta info.
        cfg (mmengine.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    if score_factor_list[0] is None:
        # e.g. Retina, FreeAnchor, etc.
        with_score_factors = False
    else:
        # e.g. FCOS, PAA, ATSS, etc.
        with_score_factors = True

    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)
    img_shape = img_meta['img_shape']
    nms_pre = cfg.get('nms_pre', -1)

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    mlvl_raw_scores = []   # MODIFIED: create list to store raw scores 
    mlvl_labels = []
    if with_score_factors:
        mlvl_score_factors = []
    else:
        mlvl_score_factors = None
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                          score_factor_list, mlvl_priors)):

        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

        dim = self.bbox_coder.encode_size
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
        if with_score_factors:
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2,
                                      0).reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            # remind that we set FG labels to [0, num_class-1]
            # since mmdet v2.0
            # BG cat_id: num_class
            scores = cls_score.softmax(-1)[:, :-1]

        # After https://github.com/open-mmlab/mmdetection/pull/6268/,
        # this operation keeps fewer bboxes under the same `nms_pre`.
        # There is no difference in performance for most models. If you
        # find a slight drop in performance, you can set a larger
        # `nms_pre` than before.
        score_thr = cfg.get('score_thr', 0)

        # MODIFIED: copy raw scores
        raw_scores = scores

        results = filter_scores_and_topk(
            scores, score_thr, nms_pre,
            dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results

        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']

        if with_score_factors:
            score_factor = score_factor[keep_idxs]

        # MODIFIED: store raw scores
        raw_scores = raw_scores[keep_idxs]

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)
        mlvl_raw_scores.append(raw_scores)  # MODIFIED: store raw scores from all levels
        mlvl_labels.append(labels)

        if with_score_factors:
            mlvl_score_factors.append(score_factor)

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = cat_boxes(mlvl_valid_priors)
    bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

    results = InstanceData()
    results.bboxes = bboxes
    results.scores = torch.cat(mlvl_scores)
    results.raw_scores = torch.cat(mlvl_raw_scores) # MODIFY: add raw_scores to results
    results.labels = torch.cat(mlvl_labels)
    if with_score_factors:
        results.score_factors = torch.cat(mlvl_score_factors)

    return self._bbox_post_process(
        results=results,
        cfg=cfg,
        rescale=rescale,
        with_nms=with_nms,
        img_meta=img_meta)