#date: 2025-09-05T16:53:30Z
#url: https://api.github.com/gists/78625123f1e6eb9d44f165b914b34b9e
#owner: https://api.github.com/users/calebrob6

import argparse
import os
from typing import Optional, Sequence, List, Tuple

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Sampler, Dataset
import numpy as np
from torch.utils.data import DataLoader
import rasterio
import rasterio.windows
import time
import math
from torch import Tensor

from src.models import BatchedDinoWrapper


from tqdm import tqdm
from sklearn.decomposition import PCA

TOKEN_STRIDE = "**********"
NUM_FEATS = 1024

def _list_dict_to_dict_list(samples):
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        samples: a list of dictionaries

    Returns:
        a dictionary of lists
    """
    collated = dict()
    for sample in samples:
        for key, value in sample.items():
            if key not in collated:
                collated[key] = []
            collated[key].append(value)
    return collated

def stack_samples(samples):
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], torch.Tensor):
            collated[key] = torch.stack(value)
    return collated

class GridGeoSampler(Sampler):
    def __init__(
        self,
        image_fns: List[List[str]],
        image_fn_indices: List[int],
        patch_size: int=256,
        stride: int=256,
    ):
        self.image_fn_indices = image_fn_indices
        self.patch_size = patch_size

        # tuples of the form (i, y, x, patch_size) that index into a CustomTileDataset
        self.indices = []
        for i in self.image_fn_indices:
            with rasterio.open(image_fns[i][0]) as f:
                height, width = f.height, f.width

            if patch_size > height and patch_size > width:
                self.indices.append((i, 0, 0, self.patch_size))
            else:
                for y in list(range(0, height - patch_size, stride)) + [
                    height - patch_size
                ]:
                    for x in list(range(0, width - patch_size, stride)) + [
                        width - patch_size
                    ]:
                        self.indices.append((i, y, x, self.patch_size))
        self.num_chips = len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self):
        return self.num_chips

class TileDataset(Dataset):
    def __init__(
        self,
        image_fns: List[List[str]],
        mask_fns: Optional[List[str]],
        transforms=None,
        sanity_check=True,
    ):
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        if mask_fns is not None:
            assert len(image_fns) == len(mask_fns)

        # Check to make sure that all the image and mask tile pairs are the same size
        # as a sanity check
        if sanity_check and mask_fns is not None:
            print("Running sanity check on dataset...")
            for image_fn, mask_fn in list(zip(image_fns, mask_fns)):
                with rasterio.open(image_fn[0]) as f:
                    image_height, image_width = f.shape
                with rasterio.open(mask_fn) as f:
                    mask_height, mask_width = f.shape
                assert image_height == mask_height
                assert image_width == mask_width

        self.transforms = transforms

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index: Tuple[int, int, int, int]):
        i, y, x, patch_size = index

        sample = {
            "y": y,
            "x": x,
        }

        window = rasterio.windows.Window(x, y, patch_size, patch_size)

        # Load imagery
        stack = []
        for j in range(len(self.image_fns[i])):
            image_fn = self.image_fns[i][j]
            with rasterio.open(image_fn) as f:
                image = f.read(window=window)
            stack.append(image)
        stack = np.concatenate(stack, axis=0)
        sample["image"] = torch.from_numpy(stack).float()

        # Load mask
        if self.mask_fns is not None:
            mask_fn = self.mask_fns[i]
            with rasterio.open(mask_fn) as f:
                mask = f.read(window=window)
            sample["mask"] = torch.from_numpy(mask).long()

        if self.transforms is not None:
            sample["image"] = self.transforms(sample["image"])

        return sample

class BatchedDinoWrapper(nn.Module):
    def __init__(self, layers: Optional[Sequence[int]] = None):
        super().__init__()
        # choose which transformer blocks to read; default: last layer only
        self.layers = list(range(24)) if layers is None else list(layers)

        # keep the backbone in self.backbone; don't wrap it yet
        self.backbone = torch.hub.load(
            "dinov3", "dinov3_vitl16", source="local",
            weights="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        ).eval()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        x: (N, 3, H, W)
        returns: (N, HW, C) features from the last selected layer
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)  # -> (1,3,H,W)
        assert x.dim() == 4 and x.size(1) == 3, "Expected (N,3,H,W)"

        feats_list = self.backbone.get_intermediate_layers(
            x, n=self.layers, reshape=True, norm=True
        )
        feats = feats_list[-1]  # take the last requested layer
        assert feats.dim() == 4 and feats.size(0) == x.size(0), f"Unexpected feats shape: {feats.shape}"

        N, C, h, w = feats.shape
        feats = feats.view(N, C, h * w).transpose(1, 2).contiguous()

        return feats

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed script arguments")
    parser.add_argument(
        "--input_fn",
        type=str,
        required=True,
        help="Path to the input file",
    )
    parser.add_argument(
        "--output_fn",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=0,
        help="GPU ids to use",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=1024,
        help="Patch size to use for inference (default: 1024)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=32,
        help="Padding to use for inference (default: 32)",
    )
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=1,
        help="Resize factor to use for inference (default: 1)",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Whether to run PCA on the features and save the first 3 components",
    )
    return parser


def main(args: argparse.Namespace):
    input_image_fn = args.input_fn
    if not os.path.exists(input_image_fn):
        raise FileNotFoundError(f"Input file {input_image_fn} does not exist.")
    if not (input_image_fn.lower().endswith(".tif") or input_image_fn.lower().endswith(".vrt")):
        raise ValueError("Input file must be a .tif or .vrt file")

    output_fn = args.output_fn
    if os.path.exists(output_fn) and not args.overwrite:
        raise FileExistsError(f"Output file {output_fn} already exists. Use --overwrite to overwrite it.")
    if os.path.exists(output_fn):
        print(f"WARNING: Output file {output_fn} already exists and will be overwritten.")

    patch_size = args.patch_size
    padding = args.padding
    assert int(math.log(patch_size, 2)) == math.log(patch_size, 2)
    stride = patch_size - padding * 2

    with rasterio.open(input_image_fn) as f:
        input_height, input_width = f.shape
        profile = f.profile
    if patch_size > input_height or patch_size > input_width:
        raise ValueError(f"Patch size {patch_size} is larger than image dimensions {input_height}x{input_width}")
    print(f"Input size: {input_height} x {input_width}")
    print(f"Using patch size {patch_size} with padding {padding} and stride {stride}")
    print(f"Starting inference with batch size {args.batch_size} on GPUs {args.gpus}")

    augs = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Resize((args.patch_size * args.resize_factor, args.patch_size * args.resize_factor)),
        transforms.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    ])

    dataset = TileDataset([[input_image_fn]], mask_fns=None, transforms=augs)

    sampler = GridGeoSampler(
        [[input_image_fn]], [0], patch_size=args.patch_size, stride=stride
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=16,
        collate_fn=stack_samples,
    )

    # Load model
    device = torch.device("cuda")
    model = BatchedDinoWrapper().to(device)
    model = nn.DataParallel(model, device_ids=args.gpus)

    output_height = "**********"
    output_width = "**********"

    print(f"Output size: {output_height} x {output_width} x {NUM_FEATS}")

    output = np.zeros((output_height, output_width, NUM_FEATS), dtype=np.float32)

    downscaled_patch_size = "**********"
    downsampled_padding_size = "**********"

    tic = time.time()
    for batch in tqdm(dataloader, desc="Running model"):
        images = batch["image"].to(device)
        x_coords = batch["x"]
        y_coords = batch["y"]
        batch_size = images.shape[0]

        with torch.inference_mode(), torch.amp.autocast('cuda'):
            features = model(images)
            features = features.cpu().numpy().reshape(batch_size, downscaled_patch_size, downscaled_patch_size, NUM_FEATS)

        for i in range(batch_size):
            height, width, _ = features[i].shape
            y = "**********"
            x = "**********"
            output[
                y + downsampled_padding_size : y + height - downsampled_padding_size, x + downsampled_padding_size : x + width - downsampled_padding_size
            ] = features[i][downsampled_padding_size:-downsampled_padding_size, downsampled_padding_size:-downsampled_padding_size]


    print(f"Finished running model in {time.time()-tic:0.2f} seconds")


    new_profile = {
        "driver": "GTiff",
        "height": output_height,
        "width": output_width,
        "count": NUM_FEATS,
        "dtype": "float32",
        "crs": profile["crs"],
        "transform": "**********"
        "compress": "lzw",
        "predictor": 3,
        "nodata": 0,
        "blockxsize": 512,
        "blockysize": 512,
        "tiled": True,
        "interleave": "pixel",
        "BIGTIFF": "YES"
    }
    tic = time.time()
    with rasterio.open(output_fn, "w", **new_profile) as f:
        f.write(output.transpose(2, 0, 1))
    print(f"Wrote output to {output_fn} in {time.time()-tic:0.2f} seconds")

    if args.pca:
        print("Running PCA on features and saving first 3 components")
        tic = time.time()
        pca = PCA(n_components=3, whiten=True)
        x_pca = pca.fit_transform(output.reshape(-1, NUM_FEATS))
        x_pca = x_pca.reshape(output_height, output_width, -1)
        x_pca = torch.from_numpy(x_pca)
        x_pca = F.sigmoid(x_pca*2.0).numpy()
        x_pca = (x_pca * 255.0).astype(np.uint8).transpose(2, 0, 1)

        print(f"PCA summed explained variance: {sum(pca.explained_variance_ratio_)}")

        pca_output_fn = output_fn.replace(".tif", "_pca.tif")
        pca_profile = new_profile.copy()
        pca_profile["count"] = 3
        pca_profile["dtype"] = "uint8"
        pca_profile["predictor"] = 2
        with rasterio.open(pca_output_fn, "w", **pca_profile) as f:
            f.write(x_pca)
        print(f"Wrote PCA output to {pca_output_fn} in {time.time()-tic:0.2f} seconds")



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    main(args)pe"] = "uint8"
        pca_profile["predictor"] = 2
        with rasterio.open(pca_output_fn, "w", **pca_profile) as f:
            f.write(x_pca)
        print(f"Wrote PCA output to {pca_output_fn} in {time.time()-tic:0.2f} seconds")



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    main(args)