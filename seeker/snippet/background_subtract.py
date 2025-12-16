#date: 2025-12-16T17:12:22Z
#url: https://api.github.com/gists/a6e890cd7822a4273e6d38b4f3f77c9c
#owner: https://api.github.com/users/NolaGreek

#!/usr/bin/env python3
"""
background_subtract.py — Fast 2D background estimation and subtraction for FITS images.

Supports sigma-clipped median, SExtractor, MMM, and grid (Background2D) methods.
Optional source masking and improved interpolation.
Outputs background-subtracted image and optional background model.
"""

import argparse
import sys
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground, SExtractorBackground, MMMBackground, BkgZoomInterpolator
from photutils.segmentation import detect_sources

def main():
    parser = argparse.ArgumentParser(
        description="Estimate and subtract 2D background from astronomical FITS images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", help="Input FITS image file")
    parser.add_argument("-o", "--output", default="bkg_subtracted.fits",
                        help="Output background-subtracted FITS file")
    parser.add_argument("--bkg-output", default=None,
                        help="Optional output background model FITS file")
    parser.add_argument("--method", choices=['median', 'sextractor', 'mmm', 'grid'], default='sextractor',
                        help="Background estimator: median (sigma-clipped), sextractor, mmm, or grid (Background2D)")
    parser.add_argument("--box-size", type=int, default=100,
                        help="Box size for grid method")
    parser.add_argument("--filter-size", type=int, default=3,
                        help="Filter size for smoothing (grid method)")
    parser.add_argument("--hdu", type=str, default="0",
                        help="HDU name or index containing image data (e.g. 0, 1, or 'SCI')")
    parser.add_argument("--mask-sources", action="store_true",
                        help="Mask detected sources before background estimation (recommended)")

    args = parser.parse_args()

    try:
        with fits.open(args.input) as hdul:
            # Handle string or int HDU
            try:
                hdu_idx = int(args.hdu)
            except ValueError:
                hdu_idx = [h.name for h in hdul].index(args.hdu.upper())
            data = hdul[hdu_idx].data.astype(np.float32)  # memory efficient
            header = hdul[hdu_idx].header.copy()

        if data is None:
            sys.exit("Error: Selected HDU has no image data.")

        mask = None
        if args.mask_sources:
            mean, median, std = sigma_clipped_stats(data, sigma=3.0)
            sources = detect_sources(data - median, threshold=5*std, npixels=10)
            if sources is not None:
                mask = sources.data.astype(bool)

        # Background estimation
        if args.method in ['median', 'sextractor', 'mmm']:
            if args.method == 'median':
                bkg_estimator = MedianBackground()
            elif args.method == 'sextractor':
                bkg_estimator = SExtractorBackground()
            else:
                bkg_estimator = MMMBackground()

            if args.method == 'grid':
                bkg = Background2D(
                    data,
                    (args.box_size, args.box_size),
                    filter_size=(args.filter_size, args.filter_size),
                    bkg_estimator=bkg_estimator,
                    interpolator=BkgZoomInterpolator(),
                    mask=mask,
                    fill_value=0.0
                )
                background = bkg.background
            else:
                # Global background
                _, background_median, _ = sigma_clipped_stats(data, sigma=3.0, mask=mask)
                background = np.full_like(data, background_median)
        else:
            bkg = Background2D(
                data,
                (args.box_size, args.box_size),
                filter_size=(args.filter_size, args.filter_size),
                bkg_estimator=MedianBackground(),
                interpolator=BkgZoomInterpolator(),
                mask=mask,
                fill_value=0.0
            )
            background = bkg.background

        # Subtract background
        subtracted = data - background

        # Save subtracted image
        new_header = header.copy()
        new_header['HISTORY'] = f"Background subtracted using {args.method} method"
        new_header['BKG_METH'] = args.method
        new_header['BKG_MED'] = float(np.median(background))
        new_header['BKG_MASK'] = args.mask_sources

        fits.writeto(args.output, subtracted, new_header, overwrite=True)

        if args.bkg_output:
            fits.writeto(args.bkg_output, background, header, overwrite=True)

        sys.stderr.write(
            f"Background subtraction complete.\n"
            f"  Method: {args.method}\n"
            f"  Median background: {np.median(background):.2f}\n"
            f"  Subtracted image saved to {args.output}\n"
        )
        if args.bkg_output:
            sys.stderr.write(f"  Background model saved to {args.bkg_output}\n")

    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == "__main__":
    main()