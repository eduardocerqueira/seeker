#date: 2023-10-27T16:48:01Z
#url: https://api.github.com/gists/04375fca6727eb2df125491c390b147c
#owner: https://api.github.com/users/PM2Ring

""" Chromostereopsis illusion
    With generated blue noise dither (void & cluster),
    Poisson disk, white noise, checks, and undithered.

    See https://mathematica.stackexchange.com/q/289992
"""

import numpy as np
from scipy import ndimage
from sage.repl.image import Image

seed = int(14159265358)
rng = np.random.default_rng(seed)

palette = bytes.fromhex('000000 ff0000 0000ff')

def scale_array(arr, rsc, csc):
    arr = np.repeat(arr, rsc, axis=0)
    arr = np.repeat(arr, csc, axis=1)
    return arr

# Make blue noise matrix, void & cluster

def FindLargestVoid(BinaryPattern,StandardDeviation):
    if(np.count_nonzero(BinaryPattern)*2>=np.size(BinaryPattern)):
        BinaryPattern=np.logical_not(BinaryPattern)
    FilteredArray=np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern,1.0,0.0)),StandardDeviation)).real
    # Find the largest void
    return np.argmin(np.where(BinaryPattern,2.0,FilteredArray))

def FindTightestCluster(BinaryPattern,StandardDeviation):
    if(np.count_nonzero(BinaryPattern)*2>=np.size(BinaryPattern)):
        BinaryPattern=np.logical_not(BinaryPattern)
    FilteredArray=np.fft.ifftn(ndimage.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern,1.0,0.0)),StandardDeviation)).real
    return np.argmax(np.where(BinaryPattern,FilteredArray,-1.0))

def GetVoidAndClusterBlueNoise(OutputShape,StandardDeviation=1.5,InitialSeedFraction=0.1):
    nRank=np.prod(OutputShape)
    # Generate the initial binary pattern with a prescribed number of ones
    nInitialOne=max(1,min(int((nRank-1)/2),int(nRank*InitialSeedFraction)))
    # Start from white noise (this is the only randomized step)
    InitialBinaryPattern=np.zeros(OutputShape,dtype=np.bool_)
    #InitialBinaryPattern.flat=np.random.permutation(np.arange(nRank))<nInitialOne
    InitialBinaryPattern.flat=rng.permutation(np.arange(nRank))<nInitialOne

    # Swap ones from tightest clusters to largest voids iteratively until convergence
    while True:
        iTightestCluster=FindTightestCluster(InitialBinaryPattern,StandardDeviation)
        InitialBinaryPattern.flat[iTightestCluster]=False
        iLargestVoid=FindLargestVoid(InitialBinaryPattern,StandardDeviation)
        if(iLargestVoid==iTightestCluster):
            InitialBinaryPattern.flat[iTightestCluster]=True
            # Nothing has changed, so we have converged
            break
        else:
            InitialBinaryPattern.flat[iLargestVoid]=True
    # Rank all pixels
    DitherArray=np.zeros(OutputShape,dtype=np.int32)
    # Phase 1: Rank minority pixels in the initial binary pattern
    BinaryPattern=np.copy(InitialBinaryPattern)
    for Rank in range(nInitialOne-1,-1,-1):
        iTightestCluster=FindTightestCluster(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iTightestCluster]=False
        DitherArray.flat[iTightestCluster]=Rank
    # Phase 2: Rank the remainder of the first half of all pixels
    BinaryPattern=InitialBinaryPattern
    for Rank in range(nInitialOne,int((nRank+1)/2)):
        iLargestVoid=FindLargestVoid(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iLargestVoid]=True
        DitherArray.flat[iLargestVoid]=Rank
    # Phase 3: Rank the last half of pixels
    for Rank in range(int((nRank+1)/2),nRank):
        iTightestCluster=FindTightestCluster(BinaryPattern,StandardDeviation)
        BinaryPattern.flat[iTightestCluster]=True
        DitherArray.flat[iTightestCluster]=Rank
    return DitherArray

def bridson_sampling(dims, radius=0.05, k=30):
    dims = np.array(dims)
    ndim = dims.size

    # For the surface sampler, all new points are almost exactly 1 radius away from at least one existing sample
    sample_radius = radius * 1.0001

    # Uniform sampling on the sphere's surface
    def hypersphere_sample(center):
        vec = rng.standard_normal(size=(k, ndim))
        return center + sample_radius * vec / np.linalg.norm(vec, axis=1)[:, None]

    # Check if there are samples closer than "squared_radius" to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True

        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)
        a = tuple([slice(lo, hi) for lo, hi in zip(indmin, indmax)])
        with np.errstate(invalid='ignore'):
            if np.any(np.sum(np.square(p - P[a]), axis=ndim) < squared_radius):
                return True

    def add_point(p):
        points.append(p)
        indices = (p / cellsize).astype(int)
        P[tuple(indices)] = p

    cellsize = radius / np.sqrt(ndim)
    gridsize = np.ceil(dims / cellsize).astype(int)
    squared_radius = radius*radius

    # Positions of cells. n-dim value for each grid cell
    P = np.full(np.append(gridsize, ndim), np.nan, dtype=np.float32)

    points = []
    add_point(rng.uniform(0.4 * dims, 0.6 * dims))
    while len(points):
        p = points.pop(rng.integers(len(points)))
        Q = hypersphere_sample(p)
        for q in Q:
            if np.all(0 <= q) and np.all(q < dims) and not in_neighborhood(q):
                add_point(q)
    return P[~np.isnan(P).any(axis=-1)]

def poisson(width, height, radius=2, k=15):
    dims = width - 1, height - 1
    pts = bridson_sampling(dims, radius=radius, k=k)
    print(pts.shape[0], "points")
    x, y = np.round(pts).astype(int).T
    v0, v1 = 0, 1
    grid = np.full((height, width), v0, dtype=bool)
    grid[y, x] = v1
    return grid

masktypes = (
  None,
  "checks",
  "whitenoise",
  "bluenoise",
  "poisson",
)

@interact
def main(rad=28, scale=7, dsize=51, masktype = Selector(masktypes), auto_update=False):
    rsq = rad * rad
    isq, osq = 300 * rsq // 784, 440 * rsq // 784
    gsize = 2*rad + 1
    gsq = gsize*gsize
    print(gsq, "cells")
    dithmax = dsize * dsize

    # Get some random bits to dither the grid
    if masktype == "whitenoise":
        # Find number of 64 bit words, using ceiling division
        rlen = -(-gsq // 64)
        #print(rlen, rlen * 64)
        mask = rng.bit_generator.random_raw(rlen).view(np.uint8)
        mask = np.unpackbits(mask)[:gsq].reshape(gsize, gsize)
    elif masktype == "bluenoise":
        dith = GetVoidAndClusterBlueNoise((dsize, dsize))
        i, j = np.ogrid[:gsize, :gsize]
        mask = dith[i % dsize, j % dsize] < dithmax * 5 // 16
    elif masktype == "poisson":
        mask = poisson(gsize, gsize, radius=2, k=35)
    elif masktype == "checks":
        dith = np.array([[0, 1], [1, 0]], dtype=bool)
        dith = scale_array(dith, 3, 3)
        i, j = np.ogrid[:gsize, :gsize]
        mask = dith[i % 6, j % 6]
    else:
        mask = np.ones((gsize, gsize), dtype=bool)

    # Get squared radial distance
    y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
    dsq = y*y + x*x

    grid = mask * ((dsq < isq).astype('u1') + 2*(dsq >= osq).astype('u1'))
    #print(grid.shape, grid.dtype)

    grid = scale_array(grid, scale, scale)
    im = Image('P', grid.shape[::-1], None)
    im.pil.frombytes(grid)
    im.pil.putpalette(palette)
    im.show()
    im.pil.save('grid.png', optimize=True)
