#date: 2022-12-15T16:48:53Z
#url: https://api.github.com/gists/ef50ac6be23e9c09437a47e94b8ef89e
#owner: https://api.github.com/users/SpecLad

#!/usr/bin/env python

import argparse
import itertools
import logging
import os

import cvat_sdk
import torch
import torch.utils.data
import torchvision.models as models

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cvat_sdk.pytorch import TaskVisionDataset, ExtractBoundingBoxes

@torch.inference_mode()
def run_testing(model, dataset, *, limit = None):
    model.eval()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=lambda batch: tuple(zip(*batch))
    )    

    metric = MeanAveragePrecision()

    for i, (images, targets) in enumerate(itertools.islice(data_loader, limit)):
        if i % 100 == 0:
            logging.info(f"Processed {i}/{len(data_loader)} samples")

        outputs = model(images)

        metric.update(outputs, targets)

    logging.info("Processed all samples, computing metric...")
    result = metric.compute()
    print(f"mAP = {result['map']:%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, metavar='N')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting...')

    model = models.detection.ssd300_vgg16(
        weights=models.detection.SSD300_VGG16_Weights.COCO_V1,
    )

    with cvat_sdk.make_client(
        'app.cvat.ai', credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS"))
    ) as client:
        logging.info('Created the client')

        testset = TaskVisionDataset(client, 39696,
            transform=models.detection.SSD300_VGG16_Weights.COCO_V1.transforms(),
            target_transform=ExtractBoundingBoxes(include_shape_types=["polygon"]),
            label_name_to_index={name: index
                for index, name in enumerate(models.detection.SSD300_VGG16_Weights.COCO_V1.meta["categories"])}
        )

        logging.info('Created the testing dataset')

        run_testing(model, testset, limit=args.limit)

if __name__ == '__main__':
    main()
