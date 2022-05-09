#date: 2022-05-09T17:04:14Z
#url: https://api.github.com/gists/dd7c0f333fb65349eefda6df0e6066c2
#owner: https://api.github.com/users/jacksonloper

import h5py
import anndata

import scipy as sp
import scipy.sparse
import pandas as pd

import logging

logger=logging.getLogger(__name__)

def _h5seurat_readmeta(f):
    dct={}

    for nm in f.keys():
        if isinstance(f[nm],h5py.Group):
            if ('levels') in f[nm] and ('values' in f[nm]):
                levels=f[nm]['levels'][:]
                values=f[nm]['values'][:]-1
                rez=np.zeros(len(values),dtype=object)
                for i,lvl in enumerate(levels):
                    rez[values==i]=lvl
                rez=rez.astype("U")
                dct[nm]=rez
            else:
                logger.info(f'skipping {nm}')
        else:
            dct[nm]=f[nm][:]
            if dct[nm].dtype.kind=='O':
                dct[nm]=dct[nm].astype("U")

    if '_index' in dct:
        idx=dct.pop('_index')
    else:
        idx=None

    return pd.DataFrame(data=dct,index=idx.astype("U"))

def _h5seurat_readcsc(f):
    data=f['counts/data'][:]
    indices=f['counts/indices'][:]
    indptr=f['counts/indptr'][:]
    shape=f['counts'].attrs['dims']
    features=f['features'][:].astype('U')
    return anndata.AnnData(
        X=sp.sparse.csc_matrix((data,indices,indptr),shape=shape).T,
        var=pd.DataFrame(index=features)
    )

class Experiment:
    def __init__(self,assays,obs):
        self.obs=obs
        self._assays=assays

    def __getitem__(self,nm):
        return self._assays[nm]

def read_h5seurat(fn,assay_names):
    with h5py.File(fn) as f:
        assays={x:_h5seurat_readcsc(f['assays/'+assay_names[x]]) for x in assay_names}
        meta=_h5seurat_readmeta(f['meta.data'])

    return Experiment(assays,meta)