#date: 2025-12-16T16:57:14Z
#url: https://api.github.com/gists/fee9cffa92739d673f4c820fc9c4aa18
#owner: https://api.github.com/users/NolaGreek

#!/usr/bin/env python3
"""
astrometry_check.py — Verify FITS WCS by matching detected sources to Gaia.

Reports matching statistics, RMS offset, and approximate rotation.
Applies Gaia proper motion correction if DATE-OBS present.
Optional output of matched catalog for plotting.
"""

import argparse
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.centroids import centroid_sources, centroid_com
from astropy.stats import sigma_clipped_stats

def main():
    parser = argparse.ArgumentParser(
        description="Verify FITS WCS by matching detected sources to Gaia reference catalog.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Input FITS image file with WCS")
    parser.add_argument("-o", "--output", default=None,
                        help="Output matched source table (ECSV recommended)")
    parser.add_argument("--hdu", type=int, default=0,
                        help="HDU index containing image data")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="Matching radius in arcsec")
    parser.add_argument("--fwhm", type=float, default=5.0,
                        help="Estimated FWHM for source detection (pixels)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Detection threshold in sigma above background")

    args = parser.parse_args()

    try:
        with fits.open(args.input) as hdul:
            data = hdul[args.hdu].data
            header = hdul[args.hdu].header
            wcs = WCS(header)
            if not wcs.has_celestial:
                sys.exit("Error: Image lacks valid celestial WCS.")

        if data is None:
            sys.exit("Error: Selected HDU has no image data.")

        # Detect sources
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=args.fwhm, threshold=args.threshold*std)
        sources = daofind(data - median)
        if sources is None or len(sources) < 10:
            sys.exit(f"Error: Only {len(sources) if sources is not None else 0} sources detected (need ≥10).")

        # Refine centroids
        x_init, y_init = sources['xcentroid'], sources['ycentroid']
        x, y = centroid_sources(data, x_init, y_init, box_size=21, centroid_func=centroid_com)

        # Convert to sky coordinates
        sky = wcs.pixel_to_world(x, y)

        # Query Gaia around field center
        center = sky[0]
        radius = np.max(sky.separation(center)) + 5*u.arcmin
        job = Gaia.cone_search_async(center, radius=radius*u.deg)
        gaia = job.get_results()

        if len(gaia) == 0:
            sys.exit("Error: No Gaia sources found in field.")

        # Apply proper motion correction if DATE-OBS present
        if 'DATE-OBS' in header:
            obs_time = Time(header['DATE-OBS'])
            gaia_coord = SkyCoord(
                ra=gaia['ra']*u.deg,
                dec=gaia['dec']*u.deg,
                pm_ra_cosdec=gaia['pmra']*u.mas/u.yr,
                pm_dec=gaia['pmdec']*u.mas/u.yr,
                obstime=Time('J2016.0'),
                frame='icrs'
            )
            gaia_current = gaia_coord.apply_space_motion(obs_time)
        else:
            gaia_current = SkyCoord(ra=gaia['ra']*u.deg, dec=gaia['dec']*u.deg)

        # Match
        inst_coord = SkyCoord(ra=sky.ra, dec=sky.dec)
        idx, sep2d, _ = inst_coord.match_to_catalog_sky(gaia_current)

        good = sep2d < args.radius * u.arcsec
        n_match = np.sum(good)
        if n_match < 10:
            sys.exit(f"Error: Only {n_match} matches within {args.radius}\" (need ≥10).")

        # Calculate offsets
        ra_offset = (sky.ra[good] - gaia_current[idx[good]].ra).arcsec
        dec_offset = (sky.dec[good] - gaia_current[idx[good]].dec).arcsec
        rms = np.sqrt(np.mean(ra_offset**2 + dec_offset**2))

        # Rough rotation estimate
        position_angle = np.arctan2(dec_offset, ra_offset)
        mean_pa = np.rad2deg(np.mean(position_angle))

        print("Astrometry Check Results")
        print("=======================")
        print(f"Sources detected : {len(sources)}")
        print(f"Gaia matches     : {n_match} (within {args.radius}\")")
        print(f"RMS offset       : {rms:.3f} arcsec")
        print(f"Mean rotation    : {mean_pa:.2f} degrees")
        print("")
        if rms < 0.5:
            print("WCS appears reliable (RMS < 0.5\").")
        else:
            print("Warning: WCS offset >0.5\" — check alignment.")

        if args.output:
            matched = Table()
            matched['x_pixel'] = x[good] + 1  # 1-based
            matched['y_pixel'] = y[good] + 1
            matched['ra_image'] = sky.ra.deg[good]
            matched['dec_image'] = sky.dec.deg[good]
            matched['ra_gaia'] = gaia_current[idx[good]].ra.deg
            matched['dec_gaia'] = gaia_current[idx[good]].dec.deg
            matched['separation_arcsec'] = sep2d.arcsec[good]
            matched['separation_arcsec'].unit = u.arcsec
            matched.write(args.output, format='ascii.ecsv', overwrite=True)
            sys.stderr.write(f"Matched catalog saved to {args.output}\n")

        sys.stderr.write("Astrometry check complete.\n")

    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == "__main__":
    main()