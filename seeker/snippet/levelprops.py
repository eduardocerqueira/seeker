#date: 2025-05-12T16:40:07Z
#url: https://api.github.com/gists/7bcd188b95dd2646bdebe8c2d4278544
#owner: https://api.github.com/users/EOakes

from astrodendro.analysis import ppv_catalog
import numpy as np


class TruncatedStructure(object):

    def __init__(self, parent, value):
        self._parent = parent
        self.value = value
        self.idx = parent.idx

    def _mask(self, subtree):
        result = self._parent.values(subtree)
        return result > self.value

    def values(self, subtree=True):
        result = self._parent.values(subtree)
        return result[self._mask(subtree)]

    def indices(self, subtree=True):
        result = self._parent.indices(subtree)
        mask = self._mask(subtree)
        result = tuple(r[mask] for r in result)
        return result


def _is_trunk(structure):
    return structure.parent is None


def levelprops(dendrogram, levels, metadata):
    result = None
    shp = (len(levels), len(dendrogram))
    for structure in dendrogram:
        for i, val in enumerate(levels):

            if (not _is_trunk(structure) and val < structure.vmin):
                continue

            if val > structure.height:
                continue

            s = TruncatedStructure(structure, val)
            rec = ppv_catalog([s], metadata)

            # prepare output array
            if result is None:
                dtype = np.array(rec).dtype
                result = np.zeros(shp, dtype=dtype)
                result[:] = np.nan
                for j in range(shp[1]):
                    result[:, j]['_idx'] = j

            # copy this record up through all (leafward) substructures
            rec = np.array(rec[0])
            for sid in [structure] + structure.descendants:
                result[i, sid.idx] = rec

    return result
