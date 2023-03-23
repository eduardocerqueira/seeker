#date: 2023-03-23T17:03:14Z
#url: https://api.github.com/gists/2216d1d51ffb615408525c90e156fc0a
#owner: https://api.github.com/users/bgpierc

# condensed version of Larkin's delentropy modification from
# https://gist.github.com/mxmlnkn/5a225478bc576044b035ad4a45c54f73

import numpy as np


def delentropy(image):
    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    if True:
        # see paper eq. (4)
        fx = ( image[:,2:] - image[:,:-2] )[1:-1,:]
        fy = ( image[2:,:] - image[:-2,:] )[:,1:-1]
    else:
        # throw away last row, because it seems to show some artifacts which it shouldn't really
        # Cleaning this up does not seem to work
        kernelDiffY = np.array( [ [-1,-1],[1,1] ] )
        fx = signal.fftconvolve( image, kernelDiffY.T ).astype( image.dtype )[:-1,:-1]
        fy = signal.fftconvolve( image, kernelDiffY   ).astype( image.dtype )[:-1,:-1]
    diffRange = np.max( [ np.abs( fx.min() ), np.abs( fx.max() ), np.abs( fy.min() ), np.abs( fy.max() ) ] )
    if diffRange >= 200   and diffRange <= 255  : diffRange = 255
    if diffRange >= 60000 and diffRange <= 65535: diffRange = 65535

    # see paper eq. (17)
    # The bin edges must be integers, that's why the number of bins and range depends on each other
    nBins = min( 1024, 2*diffRange+1 )
    if image.dtype == float:
        nBins = 1024
    # Centering the bins is necessary because else all value will lie on
    # the bin edges thereby leading to assymetric artifacts
    dbin = 0 if image.dtype == float else 0.5
    r = diffRange + dbin
    delDensity, xedges, yedges = np.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-r,r], [-r,r] ] )
    if nBins == 2*diffRange+1:
        assert( xedges[1] - xedges[0] == 1.0 )
        assert( yedges[1] - yedges[0] == 1.0 )

    # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
    # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
    #assert( np.product( np.array( image.shape ) - 1 ) == np.sum( delDensity ) )
    delDensity = delDensity / np.sum( delDensity ) # see paper eq. (17)
    delDensity = delDensity.T
    # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
    # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
    H = - 0.5 * np.sum( delDensity[ delDensity.nonzero() ] * np.log2( delDensity[ delDensity.nonzero() ] ) ) # see paper eq. (16)
    return H
