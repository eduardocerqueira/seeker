#date: 2023-03-02T16:38:59Z
#url: https://api.github.com/gists/7ab80c4f1c18461c34093b5c5ace5338
#owner: https://api.github.com/users/ivirshup

# Docs: https://docs.h5py.org/en/stable/vds.html

import h5py
import numpy as np

def virtual_concatenate(datasets: list[h5py.Dataset]) -> h5py.VirtualLayout:
    """Concatenate datasets along the first axis."""
    vds = h5py.VirtualLayout(shape=sum(d.shape[0] for d in datasets), dtype=datasets[0].dtype)
    offset = 0
    for d in datasets:
        source = h5py.VirtualSource(d)
        vds[offset:offset+d.shape[0]] = source
        offset += d.shape[0]
    return vds

with h5py.File("test_virtual.h5", "w") as f
    a = f.create_dataset("a", data=np.arange(3), chunks=None)
    b = f.create_dataset("b", data=np.arange(3, 4), chunks=None)
    c = f.create_dataset("c", data=np.arange(4, 10), chunks=None)

    expected = np.concatenate([a[:], b[:], c[:]])
    result = f.create_virtual_dataset("combined", virtual_concatenate([a, b, c]))
    np.testing.assert_equal(expected, result[:])