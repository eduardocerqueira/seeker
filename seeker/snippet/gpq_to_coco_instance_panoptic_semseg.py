#date: 2025-03-19T17:02:47Z
#url: https://api.github.com/gists/16061ee49e303914bf30b46654784547
#owner: https://api.github.com/users/fangzp

"""
Convert GeoTIFF and GeoParquet agricultural field data to COCO detection & panoptic segmentation format.
Handles both thing (agricultural fields) and stuff (background) classes.
"""

import os
from pathlib import Path
import json
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
import PIL.Image as Image
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import warnings

@dataclass
class CategoryInfo:
    """Store category information for COCO format"""
    id: int
    name: str
    supercategory: str
    isthing: bool
    color: List[int]

class InstanceInfo(NamedTuple):
    """Store instance information including ID and color"""
    id: int
    color: np.ndarray
    mask: np.ndarray
    bbox: List[float]
    area: float

def rgb2id(color: np.ndarray) -> int:
    """Convert RGB color to unique ID using panopticapi formula"""
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

class PanopticConverter:
    """Convert GeoTIFF and GeoParquet data to COCO panoptic segmentation format"""
    
    def __init__(
        self,
        categories_path: str,
        output_dir: str,
        min_area: int = 100,
        background_id: int = 4988569
    ):
        self.output_dir = Path(output_dir)
        self.min_area = min_area
        self.background_id = background_id
        self.categories = self._load_categories(categories_path)
        self._setup_directories()

    def _load_categories(self, categories_path: str) -> List[CategoryInfo]:
        """Load category definitions from JSON file"""
        with open(categories_path) as f:
            categories_data = json.load(f)
        return [CategoryInfo(**cat) for cat in categories_data]
    
    def _setup_directories(self):
        """Create necessary output directories"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "panoptic_train").mkdir(exist_ok=True)
        (self.output_dir / "panoptic_semseg_train").mkdir(exist_ok=True)
        (self.output_dir / "panoptic_test").mkdir(exist_ok=True)
        (self.output_dir / "panoptic_semseg_test").mkdir(exist_ok=True)

    def _generate_instance_color(self) -> np.ndarray:
        """Generate a unique color for an instance by adding jitter to base field color"""
        field_category = next(cat for cat in self.categories if cat.name == "ag_field")
        base_color = np.array(field_category.color)
        
        while True:
            jitter = np.random.randint(-20, 21, size=3)
            color = np.clip(base_color + jitter, 0, 255)
            
            if rgb2id(color) != self.background_id:
                return color

    def _compute_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """Compute bounding box from binary mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]

    def _create_instance(
        self,
        geom,
        transform: rio.transform.Affine,
        height: int,
        width: int
    ) -> Optional[InstanceInfo]:
        """Create instance information including mask, color, and metrics"""
        try:
            mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                dtype=np.uint8,
                all_touched=True
            )
        except Exception as e:
            warnings.warn(f"Failed to rasterize geometry: {e}")
            return None
            
        if mask is None or not mask.any():
            warnings.warn("mask is None or not mask.any()")
            warnings.warn(f'Geom was: {geom}')
            return None
            
        area = float(np.sum(mask))
        if area < self.min_area:
            warnings.warn("area below min_area threshold")
            return None
            
        color = self._generate_instance_color()
        instance_id = rgb2id(color)
        bbox = self._compute_bbox_from_mask(mask)
        
        return InstanceInfo(
            id=instance_id,
            color=color,
            mask=mask,
            bbox=bbox,
            area=area
        )

    def convert_single_image(
        self,
        geotiff_path: str,
        geoparquet_path: str,
        image_id: int,
        debug: bool = True
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[InstanceInfo]]:
        """Convert a single GeoTIFF/GeoParquet pair to COCO format"""
        if debug:
            print(f"\nProcessing {os.path.basename(geotiff_path)}")
            
        # Read raster data
        with rio.open(geotiff_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            raster_crs = src.crs
            
        # Read vector data from GeoParquet
        gdf = gpd.read_parquet(geoparquet_path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        
        # Get categories
        field_category = next(cat for cat in self.categories if cat.name == "ag_field")
        background_category = next(cat for cat in self.categories if cat.name == "background")
        
        # Create instances
        instances = []
        instance_anns = []
        panoptic_segments = []
        
        # Process each geometry
        for idx, row in gdf.iterrows():
            instance = self._create_instance(row.geometry, transform, height, width)
            if instance is None:
                if debug:
                    warnings.warn(f"Skipping instance {idx} - failed validation")
                continue
                
            if any(i.id == instance.id for i in instances):
                if debug:
                    warnings.warn(f"Duplicate instance ID {instance.id} detected - regenerating")
                continue
                
            instances.append(instance)
            
            instance_anns.append({
                "id": instance.id,
                "image_id": image_id,
                "category_id": field_category.id,
                "area": instance.area,
                "bbox": instance.bbox,
                "iscrowd": 0
            })
            
            panoptic_segments.append({
                "id": instance.id,
                "category_id": field_category.id,
                "area": instance.area,
                "bbox": instance.bbox,
                "iscrowd": 0
            })
        
        # Calculate background
        background_mask = np.ones((height, width), dtype=bool)
        for instance in instances:
            background_mask &= (instance.mask == 0)
            
        background_area = float(np.sum(background_mask))
        panoptic_segments.append({
            "id": self.background_id,
            "category_id": background_category.id,
            "area": background_area,
            "bbox": [0, 0, width, height],
            "iscrowd": 0
        })
        
        # Create image info
        image_info = {
            "file_name": os.path.basename(geotiff_path),
            "id": image_id,
            "height": height,
            "width": width
        }
        
        # Create panoptic annotation
        panoptic_ann = {
            "image_id": image_id,
            "file_name": os.path.basename(geotiff_path).replace(".tif", ".png"),
            "segments_info": panoptic_segments
        }
        
        return image_info, instance_anns, panoptic_ann, instances

    def generate_segmentation_images(
        self,
        instances: List[InstanceInfo],
        height: int,
        width: int,
        image_name: str,
        split: str
    ):
        """Generate panoptic and semantic segmentation PNG files"""
        semantic_seg = np.full((height, width), 255, dtype=np.uint8)
        panoptic_seg = np.zeros((height, width, 3), dtype=np.uint8)
        
        background_category = next(cat for cat in self.categories if cat.name == "background")
        panoptic_seg[:, :] = background_category.color
        
        for instance in instances:
            semantic_seg[instance.mask == 1] = 0
            
            for i in range(3):
                panoptic_seg[:, :, i][instance.mask == 1] = instance.color[i]
        
        base_name = image_name.replace(".tif", "")
        Image.fromarray(panoptic_seg).save(
            self.output_dir / f"panoptic_{split}" / f"{base_name}.png"
        )
        Image.fromarray(semantic_seg).save(
            self.output_dir / f"panoptic_semseg_{split}" / f"{base_name}.png"
        )

    def convert_dataset(
        self,
        geotiff_dir: str,
        geoparquet_dir: str,
        split: str = "train"
    ):
        """Convert entire dataset of GeoTIFF/GeoParquet pairs to COCO format"""
        geotiff_files = sorted(Path(geotiff_dir).glob("*.tif"))
        
        instances_dict = {
            "images": [],
            "annotations": [],
            "categories": [c.__dict__ for c in self.categories if c.isthing]
        }
        
        panoptic_dict = {
            "images": [],
            "annotations": [],
            "categories": [c.__dict__ for c in self.categories]
        }
        
        for image_id, geotiff_path in enumerate(geotiff_files):
            geoparquet_path = Path(geoparquet_dir) / f"{geotiff_path.stem}.parquet"
            if not geoparquet_path.exists():
                warnings.warn(f"No GeoParquet found for {geotiff_path}")
                continue
                
            image_info, instance_anns, panoptic_ann, instances = self.convert_single_image(
                str(geotiff_path), str(geoparquet_path), image_id
            )
            
            instances_dict["images"].append(image_info)
            instances_dict["annotations"].extend(instance_anns)
            panoptic_dict["images"].append(image_info)
            panoptic_dict["annotations"].append(panoptic_ann)
            
            self.generate_segmentation_images(
                instances,
                image_info["height"],
                image_info["width"],
                os.path.basename(geotiff_path),
                split
            )
            
        if not instances_dict["annotations"]:
            warnings.warn("No valid instances found in dataset!")
            
        instance_ids = {ann["id"] for ann in instances_dict["annotations"]}
        panoptic_ids = {
            segment["id"]
            for ann in panoptic_dict["annotations"]
            for segment in ann["segments_info"]
            if segment["id"] != self.background_id
        }
        
        if instance_ids != panoptic_ids:
            warnings.warn(
                f"Mismatch in instance IDs between formats!\n"
                f"Instances only: {instance_ids - panoptic_ids}\n"
                f"Panoptic only: {panoptic_ids - instance_ids}"
            )
            
        with open(self.output_dir / "annotations" / f"instances_{split}.json", "w") as f:
            json.dump(instances_dict, f)
            
        with open(self.output_dir / "annotations" / f"panoptic_{split}.json", "w") as f:
            json.dump(panoptic_dict, f)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF and GeoParquet data to COCO panoptic format"
    )
    parser.add_argument("--geotiff_dir", required=True, help="Directory with GeoTIFF files")
    parser.add_argument("--geoparquet_dir", required=True, help="Directory with GeoParquet files")
    parser.add_argument("--categories", required=True, help="Path to categories JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--min_area", type=int, default=100, help="Minimum instance area in pixels")
    
    args = parser.parse_args()
    
    converter = PanopticConverter(
        args.categories,
        args.output_dir,
        min_area=args.min_area
    )
    
    converter.convert_dataset(
        args.geotiff_dir,
        args.geoparquet_dir,
        args.split
    )