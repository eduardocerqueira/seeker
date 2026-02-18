#date: 2026-02-18T17:44:44Z
#url: https://api.github.com/gists/714bda96ff1ea5c5f8b41634c854fa9f
#owner: https://api.github.com/users/c7huang

# python scripts/visualize_dataloader.py machine=ch dataset=megatrain_13d_518_many_ar_36ipg_8g dataset.num_workers=0 dataset.num_views=24 dataset.sort_views=true train_params.max_num_of_imgs_per_gpu=24 model=pi3

import sys
import logging
import threading
import hydra
import numpy as np
import viser

from omegaconf import DictConfig, OmegaConf
from mapanything.datasets import get_train_data_loader
from mapanything.utils.misc import StreamToLogger
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


def main(args):
    data_loader = get_train_data_loader(
        dataset=args.dataset.train_dataset,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
        num_workers=args.dataset.num_workers,
        pin_mem=True,
        shuffle=True,
        drop_last=True,
    )

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(0)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(0)

    server = viser.ViserServer()
    next_batch_event = threading.Event()

    for batch_idx, batch in enumerate(data_loader):
        # Batch
        # list(dict_keys(
        #     [
        #         "img", (1, 3, H, W)
        #         "camera_pose", # Pose, (1, 4, 4)
        #         "camera_intrinsics", # Intrinsic, (1, 3, 3)
        #         "non_ambiguous_mask", # Non ambiguous mask, (1, H, W)
        #         "pts3d",      # Global point map, (1, H, W, 3)
        #         "valid_mask", # Valid mask, (1, H, W)
        #         "pts3d_cam",  # Camera point map, (1, H, W, 3)
        #     ]
        # ))
        # Visualization
        print(f"Loading and visualizing batch {batch_idx}: {batch[0]['dataset']}...")

        next_batch_event.clear()
        server.scene.reset()
        server.scene.set_up_direction("-y")
        server.gui.reset()
        next_btn = server.gui.add_button("Next Batch", icon="arrow-right")

        @next_btn.on_click
        def _(_):
            next_batch_event.set()

        N = len(batch)
        if N == 0:
            continue

        c2w_0 = batch[0]["camera_pose"][0].detach().cpu().numpy()  # (4, 4)
        w2c_0 = np.linalg.inv(c2w_0)

        frustum_handles = []
        ptcloud_handles = []

        for i, frame in enumerate(batch):
            img_tensor = frame["img"][0].detach().cpu().numpy()  # (3, H, W)
            img = np.transpose(img_tensor, (1, 2, 0))  # (H, W, 3)

            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

            H, W = img.shape[:2]

            intrinsics = frame["camera_intrinsics"][0].detach().cpu().numpy()  # (3, 3)
            fy = intrinsics[1, 1]
            fov = 2.0 * np.arctan(H / (2.0 * fy))

            c2w_i = frame["camera_pose"][0].detach().cpu().numpy()  # (4, 4)
            rel_c2w = w2c_0 @ c2w_i
            pos = rel_c2w[:3, 3]
            rot_mat = rel_c2w[:3, :3]

            quat_xyzw = R.from_matrix(rot_mat).as_quat()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            frustum = server.scene.add_camera_frustum(
                name=f"/cameras/cam_{i}",
                fov=fov,
                aspect=W / H,
                scale=0.1,
                image=img,
                wxyz=quat_wxyz,
                position=pos,
                visible=(i == 0),  # Only the first frame is initially visible
            )
            frustum_handles.append(frustum)

            pts3d_global = frame["pts3d"][0].detach().cpu().numpy()  # (H, W, 3)
            valid_mask = (
                frame["valid_mask"][0].detach().cpu().numpy().astype(bool)
            )  # (H, W)
            valid_pts_global = pts3d_global[valid_mask]  # (P, 3)
            valid_colors = img[valid_mask]  # (P, 3)

            if len(valid_pts_global) > 0:
                pts_homo = np.concatenate(
                    [valid_pts_global, np.ones_like(valid_pts_global[:, :1])], axis=-1
                )
                pts_rel = (w2c_0 @ pts_homo.T).T[:, :3]
            else:
                pts_rel = np.zeros((0, 3))

            ptcloud = server.scene.add_point_cloud(
                name=f"/points/pts_{i}",
                points=pts_rel,
                colors=valid_colors,
                point_size=0.01,
                visible=(i == 0),  # Only the first point cloud is initially visible
            )
            ptcloud_handles.append(ptcloud)

        frame_slider = server.gui.add_slider(
            "Frame Index", min=0, max=N - 1, step=1, initial_value=0
        )

        @frame_slider.on_update
        def _(event):
            idx = event.target.value
            for i, frustum in enumerate(frustum_handles):
                frustum.visible = i == idx
            for i, pt in enumerate(ptcloud_handles):
                pt.visible = i <= idx

        next_batch_event.wait()


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def execute_visualization(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    main(cfg)


if __name__ == "__main__":
    execute_visualization()  # noqa
