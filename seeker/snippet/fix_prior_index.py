#date: 2026-02-19T17:46:41Z
#url: https://api.github.com/gists/4aaec5328634498a3e7e45041f7024fd
#owner: https://api.github.com/users/joshua-laughner

from argparse import ArgumentParser
import netCDF4
import numpy as np


def main():
    p = ArgumentParser(description='Fix prior_index in a TCCON private netCDF file')
    p.add_argument('netcdf_file', help='The netCDF file to fix, will be adjusted in-place')

    clargs = p.parse_args()
    with netCDF4.Dataset(clargs.netcdf_file, 'a') as ds:
        if not check_prior_index(ds):
            print('Some prior indices are incorrect, they will be recalculated')
            correct_prior_index(ds, assign=True)
        else:
            print('All prior indices are correct')


def check_prior_index(ds: netCDF4.Dataset, max_delta_hours: float = 1.55) -> bool:
    """Check that the prior index values are correct

    This will check that the difference between the ZPD time and prior time for each
    spectrum, given the current prior indices, is within `max_delta_hours` hours.
    The default value of 1.55 for that value reflects that the GEOS priors change
    every 3 hours, so no spectrum should be more than half that from its prior time,
    with a small amount of padding to avoid false positives from rounding errors.
    (This is meant to catch egregiously wrong prior indexing.)

    Returns
    -------
    bool
        ``True`` if the prior indices are correct, ``False`` otherwise.
    """
    time = ds['time'][:]
    prior_time = ds['prior_time'][:]
    prior_index = ds['prior_index'][:]

    expanded_prior_time = prior_time[prior_index]
    max_dt_sec = np.ma.max(np.abs(time - expanded_prior_time))
    return max_dt_sec < (max_delta_hours * 3600)


def correct_prior_index(ds: netCDF4.Dataset, assign: bool = True) -> np.ndarray:
    """Correct the prior index values

    This will recalculate the prior index values by finding the index of the
    prior time closest to each ZPD time. Note that this should only be used if
    assigning the prior indices based on the .mav file block headers fails due
    to spectra missing from the runlog or similar reasons, as this may misassign
    prior indices very close to the transition from one prior to the next if your
    Python installation has slightly different numerics than your Fortran one.

    Parameters
    ----------
    ds
        The private netCDF dataset to calculate new prior indices for

    assign
        If ``True``, then the revised prior indices will be assigned to
        the "prior_index" variable in the netCDF file (which must exist).

    Returns
    -------
    np.ndarray
        The new prior indices
    """
    time = ds['time'][:]
    prior_time = ds['prior_time'][:]


    dt = prior_time[np.newaxis,:] - time[:,np.newaxis]
    new_indices = np.ma.argmin(np.abs(dt), axis=1)
    if assign:
        ndiff = np.sum(new_indices != ds['prior_index'][:])
        ntot = new_indices.size
        print(f'{ndiff} of {ntot} prior indices were changed')
        ds['prior_index'][:] = new_indices
    return new_indices


if __name__ == '__main__':
    main()