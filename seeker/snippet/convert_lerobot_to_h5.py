#date: 2026-01-08T17:08:09Z
#url: https://api.github.com/gists/75b61460275268f826c72132771e7e5e
#owner: https://api.github.com/users/AkashKarnatak

"""
Script lerobot to h5.
# --repo-id     Your unique repo ID on Hugging Face Hub
# --output_dir  Save path to h5 file

python unitree_lerobot/utils/convert_lerobot_to_h5.py.py \
    --repo-id your_name/g1_grabcube_double_hand \
    --output_dir "$HOME/datasets/g1_grabcube_double_hand"
"""

import cv2
import torch
import numpy as np
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import os
import cv2
import h5py
import tyro
from tqdm import tqdm
from pathlib import Path

# modify these parameters according to your config
camera_map = {
    "cam_high": "top",
    "cam_left_wrist": "wrist",
    "cam_right_wrist": "wrist",
}
rad_limits = [
    [-1.91986, 1.91986],
    [-1.74533, 1.74533],
    [-1.69, 1.69],
    [-1.65806, 1.65806],
    [-2.74385, 2.84121],
    [-0.174533, 1.74533],
]


def m100_100_to_rad(pos, rad_limits):
    def remap(x, l1, u1, l2, u2):
        return l2 + ((x - l1) / (u1 - l1)) * (u2 - l2)

    m100_100_limits = [
        [-100, 100],
        [-100, 100],
        [-100, 100],
        [-100, 100],
        [-100, 100],
        [0, 100],
    ]

    return torch.tensor(
        [remap(x, *m100_100_limits[i], *rad_limits[i]) for i, x in enumerate(pos)]
    )


class LeRobotDataProcessor:
    def __init__(
        self,
        repo_id: str,
        rad_limits: list,
        root: str = None,
        camera_map: dict = None,
        image_dtype: str = "to_unit8",
    ) -> None:
        self.image_dtype = image_dtype
        self.rad_limits = rad_limits
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)
        self.camera_map = camera_map

    def process_episode(self, episode_index: int) -> dict:
        """Process a single episode to extract camera images, state, and action."""
        from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_index]

        episode = defaultdict(list)
        cameras = defaultdict(list)

        for step_idx in tqdm(
            range(from_idx, to_idx),
            desc=f"Episode {episode_index}",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            step = self.dataset[step_idx]

            image_dict = {
                key.split(".")[2]: cv2.resize(
                    np.transpose((value.numpy() * 255).astype(np.uint8), (1, 2, 0)),
                    (640, 480),
                )  # do i really need size conversion?
                for key, value in step.items()
                if key.startswith("observation.image") and len(key.split(".")) >= 3
            }

            if self.camera_map:
                image_dict = {k: image_dict[v] for k, v in self.camera_map.items()}

            for key, value in image_dict.items():
                if self.image_dtype == "to_unit8":
                    cameras[key].append(value)
                elif self.image_dtype == "to_bytes":
                    success, encoded_img = cv2.imencode(
                        ".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
                    if not success:
                        raise ValueError(f"Image encoding failed for key: {key}")
                    cameras[key].append(np.void(encoded_img.tobytes()))

            episode["joint_positions"].append(
                m100_100_to_rad(step["observation.state"], self.rad_limits)
            )
            episode["action"].append(m100_100_to_rad(step["action"], self.rad_limits))

        for cam_name in cameras:
            cameras[cam_name] = np.stack(cameras[cam_name], axis=0)

        episode["joint_positions"] = np.stack(episode["joint_positions"], axis=0)
        episode["action"] = np.stack(episode["action"], axis=0)
        episode["cameras"] = cameras
        episode["language_raw"] = step["task"]
        episode["episode_index"] = episode_index

        return episode


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        """Write episode data to HDF5 file."""

        episode_index = episode["episode_index"]

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        T, DOF = episode["action"].shape

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as f:
            # Root-level datasets
            f.create_dataset("action", data=episode["action"])
            f.create_dataset("language_raw", data=episode["language_raw"])
            f.create_dataset(
                "substep_reasonings",
                data=np.array([f"step {i}".encode() for i in range(T)]),
                compression="gzip",
            )

            # Observations group
            obs = f.create_group("observations")

            # Images group (multi-view)
            images = obs.create_group("images")
            for cam_name, cam_value in episode["cameras"].items():
                images.create_dataset(cam_name, data=cam_value)

            # Other observation datasets
            obs.create_dataset("joint_positions", data=episode["joint_positions"])
            obs.create_dataset("qpos", data=np.zeros((T, DOF), dtype=np.float32))
            obs.create_dataset("qvel", data=np.zeros((T, DOF), dtype=np.float32))


def lerobot_to_h5(
    repo_id: str,
    output_dir: Path,
    root: str = None,
) -> None:
    """Main function to process and write LeRobot data to HDF5 format."""

    # Initialize data processor and H5 writer
    data_processor = LeRobotDataProcessor(
        repo_id, rad_limits, root, camera_map, image_dtype="to_unit8"
    )  # image_dtype Options: "to_unit8", "to_bytes"
    h5_writer = H5Writer(output_dir)

    # Process each episode
    for episode_index in tqdm(
        range(data_processor.dataset.num_episodes),
        desc="Episodes",
        position=0,
        dynamic_ncols=True,
    ):
        episode = data_processor.process_episode(episode_index)
        h5_writer.write_to_h5(episode)


if __name__ == "__main__":
    tyro.cli(lerobot_to_h5)