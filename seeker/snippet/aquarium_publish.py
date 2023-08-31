#date: 2023-08-31T17:08:17Z
#url: https://api.github.com/gists/993eac504809a19b0520c01e8a33491c
#owner: https://api.github.com/users/jhurliman

#!/usr/bin/env python3

"""Publish ground truth or inference COCO JSON labels to Aquarium."""

import argparse
import os
import typing as tp
from pathlib import Path

import aquariumlearning as al

from vpy.gt_export.coco import CocoLabels

# Datasets are uploaded to this Google Cloud Storage bucket. Example upload command:
# gsutil -m rsync -r ./carrot-20230829 gs://aquarium-customer-<redacted>/carrot-20230829
AQUARIUM_BASE_URL = "https://storage.cloud.google.com/aquarium-customer-<redacted>"


def main(api_key: str, args: tp.Any) -> None:
    # Initialize the Aquarium client
    al_client = al.Client()
    al_client.set_credentials(api_key=api_key)

    # Check that the project is valid
    datastore = args.datastore
    project_name = f"<redacted>_{datastore}"
    if not al_client.project_exists(project_name):
        raise ValueError(f"Aquarium project {project_name} does not exist")

    if not args.groundtruth_json and not args.inference_json:
        raise ValueError("Must specify --groundtruth-json or --inference-json")

    dataset_name = args.dataset_name

    if args.groundtruth_json:
        print(f'Uploading ground truth to dataset "{dataset_name}"')
        upload_groundtruth(al_client, project_name, args.groundtruth_json, dataset_name)
    if args.inference_json:
        inference_name = sanitize_id(args.inference_name) if args.inference_name else "infer"
        print(f'Uploading inference "{inference_name}" to dataset "{dataset_name}"')
        upload_inference(
            al_client, project_name, args.inference_json, dataset_name, inference_name
        )


def upload_groundtruth(
    al_client: al.Client, project_name: str, coco_labels_file: str, dataset_name: str
) -> None:
    """Upload ground truth bounding box annotations from a COCO JSON file to Aquarium."""

    # Load the COCO labels
    labels = CocoLabels.from_file(coco_labels_file)

    dataset = al_client.initialize_labeled_dataset(
        project_name=project_name, dataset_name=sanitize_id(dataset_name)
    )

    for image in labels.images:
        image_url = f"{AQUARIUM_BASE_URL}/{dataset_name}/{image.file_name}"
        frame_id = Path(image.file_name).stem
        al_image = al.Image(id="default", image_url=image_url)

        frame_id_parts = frame_id.split("_")
        # Remove _x_y_w_h tile suffix to get `datastore.source_ref`
        source_ref = "_".join(frame_id_parts[:-4]) if len(frame_id_parts) > 4 else frame_id

        labels_to_add: tp.List[al.labels.Label] = []
        for i, ann in enumerate(labels.annotations_by_image_id[image.id]):
            label_id = f"{frame_id}_gt_{i}"
            classification = ann.category_name(labels.categories)
            x, y, w, h = ann.bbox

            label = al.Bbox2DLabel(
                id=label_id,
                classification=classification,
                top=y,
                left=x,
                width=w,
                height=h,
                user_attrs={"source_ref": source_ref},
            )
            labels_to_add.append(label)

        dataset.create_and_add_labeled_frame(
            frame_id=frame_id, sensor_data=[al_image], labels=labels_to_add, user_metadata=[]
        )

    al_client.create_or_update_labeled_dataset(dataset=dataset)


def upload_inference(
    al_client: al.Client,
    project_name: str,
    coco_labels_file: str,
    dataset_name: str,
    inference_name: str,
) -> None:
    """Upload inference bounding box annotations from a COCO JSON file to Aquarium."""

    if inference_name == "gt":
        raise ValueError(
            "Cannot use inference_name=gt, as this is reserved for ground truth labels"
        )

    # Load the COCO labels
    labels = CocoLabels.from_file(coco_labels_file)

    inference_set = al_client.initialize_inference_set(
        project_name=project_name,
        base_dataset_name=sanitize_id(dataset_name),
        inference_set_name=inference_name,
    )

    for image in labels.images:
        frame_id = Path(image.file_name).stem
        gt_anns = labels.annotations_by_image_id[image.id]
        anns = labels.annotations_by_image_id[image.id]

        frame_id_parts = frame_id.split("_")
        # Remove _x_y_w_h tile suffix to get `datastore.source_ref`
        source_ref = "_".join(frame_id_parts[:-4]) if len(frame_id_parts) > 4 else frame_id

        inferences_to_add: tp.List[al.inferences.Inference] = []
        for i, ann in enumerate(anns):
            label_id = f"{frame_id}_{inference_name}_{i}"
            classification = ann.category_name(labels.categories)
            x, y, w, h = ann.bbox

            # Check if the refined annotation exactly matches the ground truth
            gt_x, gt_y, gt_w, gt_h = gt_anns[i].bbox
            matches_gt = x == gt_x and y == gt_y and w == gt_w and h == gt_h

            inference = al.Bbox2DInference(
                id=label_id,
                classification=classification,
                confidence=1.0,
                top=y,
                left=x,
                width=w,
                height=h,
                user_attrs={"matches_gt": matches_gt, "area": ann.area},
            )
            inferences_to_add.append(inference)

        inference_set.create_and_add_inference_frame(
            frame_id=frame_id,
            inferences=inferences_to_add,
            user_metadata=[al.UserMetadataEntry("source_ref", source_ref)],
        )

    al_client.create_or_update_inference_set(inference_set=inference_set)


def sanitize_id(id: str) -> str:
    """Only allow alphanumeric and underscore characters in IDs. Replace all other characters with
    underscores."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in id)


if __name__ == "__main__":
    # Usage: AQUARIUM_API_KEY=... ./aquarium_publish.py
    #   --datastore <crop>
    #   --dataset-name <name>
    #   [--groundtruth-json <test.json>]
    #   [--inference-name <name>]
    #   [--inference-json <test.json>]
    parser = argparse.ArgumentParser(
        usage="AQUARIUM_API_KEY=... ./aquarium_publish.py",
        description="Publish ground truth or inference COCO JSON labels to Aquarium.",
    )
    parser.add_argument("--datastore", required=True, help="Datastore name (ex: carrot)")
    parser.add_argument("--dataset-name", required=True, help="Dataset name (ex: carrot-20230829)")
    parser.add_argument(
        "--groundtruth-json", help="Ground truth COCO JSON file (ex: ./carrot-20230829/test.json)"
    )
    parser.add_argument("--inference-name", help="Inference name (ex: infer)")
    parser.add_argument(
        "--inference-json",
        help="Inference COCO JSON file (ex: ./carrot-20230829/results-test.json)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("AQUARIUM_API_KEY")
    if not api_key:
        raise ValueError("AQUARIUM_API_KEY environment variable must be set")

    main(api_key, args)
