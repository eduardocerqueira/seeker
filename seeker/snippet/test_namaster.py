#date: 2024-09-13T16:55:11Z
#url: https://api.github.com/gists/7fc4d1fec9b2752851f2e498e620e1a1
#owner: https://api.github.com/users/ajouellette

import os                                                                                                                                                                                                            
import numpy as np
import healpy as hp
import pymaster as nmt 
import joblib


def get_workspace(nmt_field1, nmt_field2, nmt_bins, wksp_cache=None):
    if wksp_cache is None:
        print("Computing workspace")
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        return wksp

    hash_key = joblib.hash([nmt_field1.get_mask(), nmt_field1.spin, nmt_field2.get_mask(), nmt_field2.spin])
    wksp_file = f"{wksp_cache}/cl/{hash_key}.fits"

    try:
        # load from existing file
        wksp = nmt.NmtWorkspace.from_file(wksp_file)
        wksp.check_unbinned()
        print("Using cached workspace")
        # update bins and beams after loading
        wksp.update_beams(nmt_field1.beam, nmt_field2.beam)
        wksp.update_bins(nmt_bins)
    except RuntimeError:
        # compute and save to file
        print("Computing workspace and saving")
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        os.makedirs(f"{wksp_cache}/cl", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b, wksp_cache=None):
    if wksp_cache is None:
        print("Computing workspace")
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        return wksp

    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field1a.spin, nmt_field2a.get_mask(), nmt_field2a.spin,
                            nmt_field1b.get_mask(), nmt_field1b.spin, nmt_field2b.get_mask(), nmt_field2b.spin])
    wksp_file = f"{wksp_cache}/cov/{hash_key}.fits"

    try:
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
        print("Using cached workspace")
    except RuntimeError:
        print("Computing workspace and saving")
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        os.makedirs(f"{wksp_cache}/cov", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


nside = 1024
ell = np.arange(3*nside)

bins = nmt.NmtBin.from_nside_linear(nside, 100)
ell_eff = bins.get_effective_ells()

# load maps
mask = hp.read_map("test_mask.fits")
shear_maps = hp.read_map("test_shear.fits", field=None)
kappa_map = hp.read_map("test_kappa.fits")

print("creating fields")
shear_field = nmt.NmtField(mask, shear_maps, spin=2)
kappa_field = nmt.NmtField(mask, [kappa_map], spin=0)

def run_analysis(field1, field2, bins, wksp_cache=None):
    print("computing cross-Cl")
    wksp = get_workspace(field1, field2, bins, wksp_cache=wksp_cache)
    pcl = nmt.compute_coupled_cell(field1, field2)
    cl = wksp.decouple_cell(pcl)

    print("computing covariance")
    cov_wksp = get_cov_workspace(field1, field2, field1, field2, wksp_cache=wksp_cache)
    pcl1 = nmt.compute_coupled_cell(field1, field1) / np.mean(field1.get_mask()**2)
    pcl2 = nmt.compute_coupled_cell(field2, field2) / np.mean(field2.get_mask()**2)
    cov = nmt.gaussian_covariance(cov_wksp, 0, 2, 0, 2, pcl1, *2*[pcl / np.mean(field1.get_mask() * field2.get_mask())],
                                  pcl2, wksp)
    return cl, cov

# run without caching
print("Running without cache")
cl, cov = run_analysis(kappa_field, shear_field, bins)
print(cl[0])
print("NaNs:", np.isnan(cov).any())
cov = cov.reshape((len(ell_eff), 2, len(ell_eff), 2))
print(np.sqrt(np.diag(cov[:,0,:,0])))
print()

# run with caching
print("Running with cache (1)")
cl, cov = run_analysis(kappa_field, shear_field, bins, wksp_cache="/home/aaronjo2/scratch/test")
print(cl[0])
print("NaNs:", np.isnan(cov).any())
cov = cov.reshape((len(ell_eff), 2, len(ell_eff), 2))
print(np.sqrt(np.diag(cov[:,0,:,0])))
print()

# re-run with cached workspaces
print("Running with cache (2)")
cl, cov = run_analysis(kappa_field, shear_field, bins, wksp_cache="/home/aaronjo2/scratch/test")
print(cl[0])
print("NaNs:", np.isnan(cov).any())
cov = cov.reshape((len(ell_eff), 2, len(ell_eff), 2))
print(np.sqrt(np.diag(cov[:,0,:,0])))