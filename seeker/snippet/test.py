#date: 2025-05-12T16:40:07Z
#url: https://api.github.com/gists/7bcd188b95dd2646bdebe8c2d4278544
#owner: https://api.github.com/users/EOakes

from astrodendro import Dendrogram
from levelprops import levelprops
from astropy.io import fits
from astropy import units as u
import numpy as np


def dendro():
    data = fits.getdata('L1448.13co.un.fits')
    data = np.nan_to_num(data)
    data = np.maximum(data, 0)

    data = data[::2, ::2, ::2]
    dg = Dendrogram.compute(data, min_value=4, min_npix=15, min_delta=0.5)

    metadata = {}
    metadata['data_unit'] = u.Jy
    metadata['beam_major'] = 22.9 * u.arcsec
    metadata['beam_minor'] = 22.9 * u.arcsec
    metadata['velocity_scale'] = 1 * u.km / u.s

    print 'Dendrogram properties'
    print '#structes:', len(dg)

    levels = np.linspace(4, 8, 5)
    lp = levelprops(dg, levels, metadata)

    return dg, lp, levels


def test_indexing():
    dg, lp, levels = dendro()

    for i, s in enumerate(dg):

        actual = lp[:, s.idx]['flux']

        v = s.values()
        expected = np.array([(v * (v > l)).sum()
                             for l in levels])

        expected[expected == 0] = np.nan

        expected[s.height < levels] = np.nan
        actual[s.height < levels] = np.nan

        expected[s.vmin > levels] = np.nan
        actual[s.vmin > levels] = np.nan

        np.testing.assert_allclose(expected, actual, rtol=1e-3)


def test_monotonic():
    dg, lp, levels = dendro()
    for s in dg:
        l = lp[:, s.idx]['flux']
        l = l[np.isfinite(l)]
        np.testing.assert_array_equal(np.sort(l), l[::-1])
