#date: 2023-03-31T17:08:22Z
#url: https://api.github.com/gists/9c71630725dab87576baa5b891a63615
#owner: https://api.github.com/users/plouvart

from PIL import Image
from pathlib import Path
import numpy as np
import sys


def adjust(
	src_f1: Path,
	tgt_f2: Path,
	out_f: Path,
) -> None:
    """Adjust colorimetry
    
    This script adjusts the colorimetry of one image
    based on the colorimetry of another.
    It does so by computing the transformation between
    the gaussian kernels of the source and target images.
    A mutlivariate Normal distribution is used, so
    the end result is more accurate than if two different
    normal distributions were used on the x and y axis.
    """
    i1 = np.array(Image.open(src_f1))
    i2 = np.array(Image.open(tgt_f2))
    p1 = i1.reshape(-1, i1.shape[-1])
    p2 = i2.reshape(-1, i2.shape[-1])
    m1 = np.mean(p1, axis=0)
    m2 = np.mean(p2, axis=0)
    e1,v1 = np.linalg.eig(np.cov(p1.T))
    e2,v2 = np.linalg.eig(np.cov(p2.T))
    # The vector from the basis resulting from the eigenvalues process may
    # not be oriented in the same direction
    # This operation reorient them correctly
    v1f = (v1 * np.sign(np.sum(v1 * v2, axis=0))).T

    Image.fromarray(
        ((v2 @ ((v1f @ (p1 - m1).T).T / e1**.5 * e2**.5).T).T + m2).reshape(*i1.shape).astype(np.uint8)
    ).save(out_f)


if __name__ == "__main__":
    adjust(
        src_f1 = Path(sys.argv[1]),
        tgt_f2 = Path(sys.argv[2]),
        out_f = Path(sys.argv[3]),
    )
	