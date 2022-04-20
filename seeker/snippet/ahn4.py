#date: 2022-04-20T17:17:06Z
#url: https://api.github.com/gists/579576d10db3c33f508fa1c74d61f0e1
#owner: https://api.github.com/users/breinbaas

import urllib.request
from tifffile import imread
import numpy as np


class Tile:
    def __init__(self):
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.data = None

    @classmethod
    def from_ahn4(cls, xmin: int, ymin: int, xmax: int, ymax: int) -> 'Tile':
        result = Tile()
        size_x = (xmax - xmin) * 2
        size_y = (ymax - ymin) * 2
        urllib.request.urlretrieve(
            f"https://ahn.arcgisonline.nl/arcgis/rest/services/Hoogtebestand/AHN4_DTM_50cm/ImageServer/exportImage?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR=&size={size_x},{size_y}&imageSR=&time=&format=tiff&pixelType=F64&noData=&noDataInterpretation=esriNoDataMatchAny&interpolation=+RSP_BilinearInterpolation&compression=&compressionQuality=&bandIds=&mosaicRule=&renderingRule=&f=image",
            "data.tiff",
        )
        result.xmin = xmin
        result.ymax = ymax
        result.data = imread("data.tiff")
        result.xmax = xmin + int(result.data.shape[1] * 0.5)
        result.ymin = ymax - int(result.data.shape[0] * 0.5)
        return result

    def get_z(self, x: float, y: float) -> float:
        if self.xmin <= x and x < self.xmax and self.ymin <= y and y <= self.ymax:
            idx = int((x - self.xmin) / 0.5)
            idy = int((self.ymax - y) / 0.5)
            return self.data[idy, idx]
        else:
            return np.nan


if __name__ == "__main__":
    ahn4tile = Tile.from_ahn4(131000, 476400, 131300, 476750)
    assert round(ahn4tile.get_z(131178.7, 476558.84), 4) == -0.0228
    assert round(ahn4tile.get_z(131178.47, 476558.79), 4) == 0.0572
    assert round(ahn4tile.get_z(131178.76, 476559.03), 4) == 0.1372
    assert round(ahn4tile.get_z(131179.02, 476558.98), 4) == -0.1028
