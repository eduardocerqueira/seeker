#date: 2022-10-03T17:27:28Z
#url: https://api.github.com/gists/b1a461c0f269a7c8ae57dcf702b0da4c
#owner: https://api.github.com/users/farukcankaya

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Instances
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels


class VisualizationHook(DatasetEvaluator):
    def __init__(self, dataset_name, storage=None, instance_mode=ColorMode.IMAGE, max_num_of_images=4):
        self.metadata = MetadataCatalog.get(dataset_name)
        self.colors = [tuple([i / 255 for i in c]) for c in self.metadata.thing_colors]
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2")
        self.storage = storage
        self.instance_mode = instance_mode
        self.max_num_of_images = max_num_of_images

    def init_vis(self):
        self.preds = {
            'GT': [],
            'P50': [],
            'P30': [],
            'P20': []
        }

    def reset(self):
        self.is_called_in_evaluation = False

    def process(self, inputs, outputs):
        self.init_vis()
        for i in range(0, min(len(inputs), self.max_num_of_images)):
            alpha = 0.4
            # Final Input (which is fed into the model)
            model_input_image = inputs[i]["image"].numpy().transpose(1, 2, 0)

            # Predictions
            self.predictions(inputs[i], outputs[i], model_input_image, alpha)

        self.add_left_labels([self.preds])

        vis_pred = np.concatenate(
            ([np.concatenate((self.preds[key]), axis=1) for key in self.preds if self.preds[key]]), axis=0)
        vis_name = f"Rows: {','.join([key for key in self.preds.keys() if self.preds[key]])}"
        if comm.is_main_process():
            self.storage.put_image(vis_name, vis_pred.transpose(2, 0, 1))

    def predictions(self, input, output, model_input_image, alpha=0.4):
        ## Ground Truth
        if "instances" in input and input["instances"].has('gt_masks'):
            assigned_colors = []
            labels = []
            for j in input["instances"].gt_classes:
                assigned_colors.append(self.colors[j])
                labels.append(self.metadata.thing_classes[j])
            visualizer = Visualizer(model_input_image, self.metadata)

            vis_gt = visualizer.overlay_instances(masks=input["instances"].gt_masks,
                                                  assigned_colors=assigned_colors,
                                                  labels=labels,
                                                  alpha=alpha)
            self.preds['GT'].append(vis_gt.get_image())
        else:
            vis_gt = Visualizer(model_input_image).get_output()
            self.preds['GT'].append(vis_gt.get_image())

        predictions = output["instances"].to(self._cpu_device)
        visualizer = Visualizer(model_input_image, self.metadata, instance_mode=self.instance_mode)
        vis_output_p20 = self.draw_instance_predictions(visualizer,
                                                        predictions=self.filter_by_threshold(predictions, 0.2),
                                                        alpha=alpha)
        self.preds['P20'].append(vis_output_p20.get_image())

        visualizer = Visualizer(model_input_image, self.metadata, instance_mode=self.instance_mode)
        vis_output_p30 = self.draw_instance_predictions(visualizer,
                                                        predictions=self.filter_by_threshold(predictions, 0.3),
                                                        alpha=alpha)
        self.preds['P30'].append(vis_output_p30.get_image())

        visualizer = Visualizer(model_input_image, self.metadata, instance_mode=self.instance_mode)
        vis_output_p50 = self.draw_instance_predictions(visualizer,
                                                        predictions=self.filter_by_threshold(predictions, 0.5),
                                                        alpha=alpha)
        self.preds['P50'].append(vis_output_p50.get_image())

    def draw_instance_predictions(self, visualizer, predictions, alpha):
        """
        Copied from utils.visualizer.py

        :param visualizer:
        :param predictions:
        :return:
        """
        # TODO: use the same color with GT
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, visualizer.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, visualizer.output.height, visualizer.output.width) for x in masks]
        else:
            masks = None

        if visualizer.metadata.get("thing_colors"):
            colors = [
                visualizer._jitter([x / 255 for x in visualizer.metadata.thing_colors[c]]) for c in classes
            ]

        if visualizer._instance_mode == ColorMode.IMAGE_BW:
            visualizer.output.reset_image(
                visualizer._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )

        visualizer.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            # keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return visualizer.output

    def add_left_labels(self, image_batch: list):
        for augs in image_batch:
            for key in augs:
                for i, image in enumerate(augs[key]):
                    augs[key][i] = self.add_left_label(image, key)

    def add_left_label(self, image: np.ndarray, label_text: str):
        from PIL import ImageFont
        if isinstance(image, np.ndarray):
            W, _, _ = image.shape
        else:
            _, W = image.size

        label_height = 20
        label_background = (255, 255, 255)  # white
        label_color = (0, 0, 0)  # black
        font_size = 18
        txt = Image.new('RGB', (W, label_height), label_background)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/DejaVuSans.ttf", font_size)
        d = ImageDraw.Draw(txt)
        w, h = d.textsize(label_text, font=fnt)
        d.text(((W - w) / 2, 0), label_text, font=fnt, fill=label_color)

        return np.concatenate(([np.asarray(txt.rotate(90, expand=True)), image]), axis=1)

    def filter_by_threshold(self, predictions, threshold):
        indices = torch.where(predictions.scores.detach() > threshold)
        ret = Instances(predictions._image_size)
        for k, v in predictions._fields.items():
            ret.set(k, v[indices])
        return ret
