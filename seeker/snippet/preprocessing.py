#date: 2024-04-12T16:57:18Z
#url: https://api.github.com/gists/4aa3eb54ec3ce4ebeafac44468b61a1a
#owner: https://api.github.com/users/Sierra-MC

"""
Preprocessing steps for the GOES LUN Moon image extraction pipeline. We will
most likely not put any of this stuff directly in the Notebook -- this is
material to summarize and/or stick in a gist.
"""

from functools import reduce
from itertools import chain, product
from operator import add, and_, gt, lt
from pathlib import Path
import re
from typing import Callable, Collection, Literal, TypeAlias

from dustgoggles.dynamic import exc_report
from dustgoggles.structures import MaybePool
import astropy.time as at
from lhorizon import LHorizon
from marslab.imgops.imgstats import ravel_valid
from more_itertools import chunked, divide
from netCDF4 import Dataset as CDFD
import numpy as np
import pandas as pd
from scipy import stats


PRODUCT_ROOT = Path("/datascratch/goes_subset/")
"""Configuration constant. Where'd you put the GOES LUN files?"""

LUNPAT = re.compile(
    r"LUN-M(?P<mode>\d)C(?P<band>\d+)"
    r"_G(?P<orb>1\d)_s(?P<year>\d{4})"
    r"(?P<day>\d{3})(?P<h>\d{2})"
    r"(?P<m>\d{2})(?P<s>\d{2})"
    r"(?P<ms>\d)"
)
"""
Regex pattern with named capture groups for GOES LUN NetCDF files.

capture groups are:
* mode: ABI observing mode (3 or 6 in this set)
* band: ABI band number (1 [blue], 3 ["veggie"], or 5 ["snow/ice"] in this set)
* orb: GOES satellite number (16, 17, or 18 in this set)
* year: 4-digit year
* day: 3-digit DOY
* h: 2-digit hour (time appears to be UTC)
* m: 2-digit minute
* s: 2-digit second
* ms: 1-digit millisecond (really decisecond but whatever, we multiply by 100)

NOTE: could probably be generalized to all standard GOES NetCDF files. 
"""

ABIFrame: TypeAlias = str
"""
String of the form 't{time}s{swath}', where 'time' and 'swath' are 0 or 1 and 
represent, respectively, indices to the first and second axes of a 4-D array.
Used as references from metadata and 2D ndarrays to the time and swath planes 
of GOES ABI radiance / radiance flag arrays. E.g., if `radiance` is the 
original ABI radiance ndarray, then the string "t0s1" denotes a reference to 
the 2-D ndarray `radiance[0, 1, :, :]`. 
"""


def make_pframe():
    """
    Make a product information DataFrame from all GOES NetCDF files in
    PRODUCT_ROOT and write it as CSV.

    Expected columns of the CSV file are:
    * fn: filename relative to PRODUCT_ROOT
    * mode, band, orb: see LUNPAT docstring; not directly relevant to this
      pipeline
    * start: image capture start time, UTC
    * jd: image capture start time, JD (not further used)
    * jdround: image capture start time, JD, rounded to 2 decimal places
      (not further used)
    * ill: %illumination of lunar disk
    * illdef: defect of illumination of lunar disk (not used)
    * brt: surface brightness of lunar disk in visual magnitudes per square
      arcsecond (not used)

    Note that ill, illdef, and brt are referenced to a geocenter -> lunar
    surface intercept point vector.
    """
    frecs = [
        {"fn": p.name} | LUNPAT.search(p.name).groupdict()
        for p in PRODUCT_ROOT.iterdir()
    ]
    pframe = pd.DataFrame(frecs)
    pframe = pframe.astype({c: "u2" for c in pframe.columns if c != "fn"})
    pframe["ms"] *= 100  # this is actually tenths of a second in the filename
    # we offset by 1970 in order to create a timedelta object, because pandas
    # expects Unix epoch time
    years = (pframe["year"] - 1970).astype("datetime64[Y]")
    # DOY is 1-indexed
    pframe["day"] = pframe["day"] - 1
    deltas = [
        pd.to_timedelta(pframe[c], unit=c)
        for c in pframe.columns
        if c not in ("fn", "mode", "band", "orb", "year")
    ]
    # this should now be a pandas Timestamp for start time UTC
    pframe["start"] = years + reduce(add, deltas)
    pframe = pframe.drop(columns=["year", "day", "h", "m", "s", "ms"])
    pframe = pframe.sort_values(by=["orb", "start"]).reset_index(drop=True)
    times = at.Time(pframe["start"])
    # convert to Julian day number to make chunking times easier
    pframe["jd"] = times.jd
    # 'jdround' chunks observations to hundredths of a day. This reduces
    # the number of calls to Horizons we need to make in the next step
    pframe["jdround"] = np.round(pframe["jd"], 2)
    jduniq, jdinv = np.unique(pframe["jdround"], return_inverse=True)
    # in chunks of 110 bins, get magnitude, surface brightness, defect of
    # illumination, % illuminated for each time bin. 110 is just about how
    # many times we can pass to Horizons at once, given the other query
    # parameters, before the URL gets too long and the API rejects it.
    # our subset of files should only actually have
    lhframes = [
        LHorizon(
            target=301, origin="500@399", epochs=chunk, quantities=(9, 10, 11)
        ).table()
        for chunk in chunked(jduniq, 110)
    ]
    lhframe = pd.concat(lhframes)
    lhframe = lhframe.reset_index(drop=True).rename(columns={"jd": "jdround"})
    assert (jduniq == lhframe["jdround"]).all()  # sanity check
    pframe.merge(lhframe, on="jdround").drop(columns="time").to_csv(
        "pframe.csv", index=None
    )


def dissect_radiance(ncfile: Path) -> dict[ABIFrame, np.ma.MaskedArray]:
    """
    Read a GOES ABI NetCDF file into a dict of MaskedArrays, one value per
    distinct "frame," defined by unique time/swath indices (first and second
    array axes). There are always 2 swaths and 1 or 2 times, so this dict will
    have either 2 or 4 items. The values and masks of the returned
    MaskedArrays are taken from the corresponding 2-D slices of the "radiance"
    and "radiance_dqf" objects respectively (all elements with any nonzero
    data quality flag are masked).
    """
    ds = CDFD(ncfile)
    rad = ds.variables["radiance"].__array__()
    dqf = ds.variables["radiance_dqf"].__array__()
    rad.mask = np.logical_or(rad.mask, dqf != 0)
    return {
        f"t{t}s{s}": rad[t, s, :, :]
        for t, s in product(range(rad.shape[0]), range(rad.shape[1]))
    }


def radstats(ncpath: Path) -> tuple[list[dict], list[dict]]:
    """
    Compute descriptive statistics for each 2-D "frame" of a GOES ABI image.
    Returns a 2-tuple whose elements are:
    * recs: list of dicts containing stats and source metadata per plane
    * failures: list of dicts containing exception information for any planes
      we couldn't successfully load / compute stats on
    """
    planes = dissect_radiance(ncpath)
    recs, failures = [], []
    for k, v in planes.items():
        baserec = {"fn": ncpath.name, "tband": int(k[1]), "swath": int(k[3])}
        try:
            rv = ravel_valid(v)
            if rv.size == 0:
                recs.append({"n_valid": rv.size} | baserec)
                continue
            recs.append(
                {
                    "kurt": stats.kurtosis(rv),
                    "skew": stats.skew(rv),
                    "n_valid": rv.size,
                    "vratio": rv.size / v.size,
                    "mean": rv.mean(),
                    "std": rv.std(),
                    "centiles": np.percentile(rv, (1, 10, 25, 75, 90, 99)),
                    "median": np.median(rv),
                    "max": rv.max(),
                }
                | baserec
            )
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            failures.append(baserec | {"ex": exc_report(ex)})
    return recs, failures


def map_radstats(ncfiles: Collection[Path]) -> tuple[list[dict], list[dict]]:
    """
    Utility function: run `radstats()` on each element of `ncfiles`. To be
    mapped across a large number of input files in parallel.
    """
    recs, failures = [], []
    for i, n in enumerate(ncfiles):
        r0, f0 = radstats(n)
        recs += r0
        failures += f0
        if i % 100 == 50:
            print(f"{i}/{len(ncfiles)}")
    return recs, failures


def make_statframe(pframe, n_threads: int = 14):
    """
    Construct a DataFrame of descriptive statistics for all products in
    `pframe` (reading files and computing statistics in `n_threads` parallel
    processes) and write it as CSV.

    This is an intermediate product.
    """
    chunks = divide(n_threads, [PRODUCT_ROOT / p for p in pframe["fn"]])
    argrecs = [{"args": (tuple(chunk),)} for chunk in chunks]
    argrecs = [r for i, r in enumerate(argrecs)]
    pool = MaybePool(n_threads)
    pool.map(map_radstats, argrecs)
    pool.close()
    pool.join()
    results = pool.get()
    pool.terminate()
    recs = tuple(chain(*[r[0] for r in results.values()]))
    fails = tuple(chain(*[r[1] for r in results.values()]))
    assert len(fails) == 0
    del results
    statframe = pd.DataFrame(recs)
    crecs = []
    cex = statframe["centiles"].explode()
    for ix, vals in cex.groupby(cex.index):
        crec = {}
        for centile, val in zip((1, 10, 25, 75, 90, 99), vals):
            crec[f"c{centile}"] = val
        crecs.append(crec)
    centframe = pd.DataFrame(crecs)
    statframe = pd.concat([statframe, centframe], axis=1)
    statframe = statframe.drop(columns="centiles")
    precol = ("fn", "tband", "swath")
    postcol = sorted(set(statframe.columns).difference(precol))
    statframe = statframe.reindex(columns=[*precol, *postcol])
    statframe["npskew"] = (
        abs(statframe["mean"] - statframe["median"]) / statframe["std"]
    )
    statframe.to_csv("statframe.csv", index=None)


def radload(
    ncfile: Path, tband: Literal[0, 1], swath: Literal[0, 1]
) -> np.ma.MaskedArray:
    """
    Shorthand function: load a specific 2D "frame" from a GOES ABI file with
    the data quality mask properly applied.
    """
    return dissect_radiance(ncfile)[f"t{tband}s{swath}"]


def make_statpred(
    df: pd.DataFrame,
    statspecs: Collection[
        tuple[str, Callable[[pd.Series, float], pd.Series], float]
    ],
) -> pd.Series:
    """
    Construct a boolean Series giving the logical conjunction of a collection
    of "predicate" operators applied to named numeric columns of a DataFrame.
    """
    return reduce(and_, (op(df[name], val) for name, op, val in statspecs))


MOONSPEC = (
    ("skew", gt, 1.3),
    ("skew", lt, 8.5),
    ("npskew", lt, 0.7),
    ("c99", gt, 5),
    ("c99", lt, 225),
)
"""Statistical heuristic: "there's probably a Moon in this GOES LUN frame"."""


def pick_moon_candidates(
    moonspec: Collection[
        tuple[str, Callable[[pd.Series, float], pd.Series], float]
    ]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge the CSV files created by `make_statframe()` and
    `make_pframe()` and perform a coarse statistical cut on the image frames.
    Return two DataFrames: the first describes image frames that probably
    contain the Moon; the second, frames that probably don't.
    """
    statframe, pframe = map(pd.read_csv, ("statframe.csv", "pframe.csv"))
    statframe = pd.merge(
        statframe, pframe[["fn", "orb", "ill", "start", "jd"]], on="fn"
    )
    moonpred = make_statpred(statframe, moonspec)
    maybe = statframe.loc[moonpred].copy()
    maybenot = statframe.loc[~moonpred].copy()
    return maybe, maybenot
