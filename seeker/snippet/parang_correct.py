#date: 2022-05-03T17:18:08Z
#url: https://api.github.com/gists/4d15ac8df92923aba390ffb9268f94dd
#owner: https://api.github.com/users/bennahugo

#!/usr/bin/env python3
import ephem
import numpy as np
import datetime
import argparse
import logging
from pyrap.tables import table as tbl
from pyrap.tables import taql
from pyrap.quanta import quantity
from astropy.coordinates import SkyCoord
from astropy import units
import pytz

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("Parangle corrector")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt

log, log_console_handler, log_formatter = create_logger()

parser = argparse.ArgumentParser(description="Parallactic corrector for MeerKAT")
parser.add_argument("ms", type=str, help="Database to correct")
parser.add_argument("--field", "-f", dest="field", type=int, default=0, help="Field index to correct")
parser.add_argument("--specialEphem", "-se", dest="ephem", default=None, type=str, help="Use special ephemeris body as defined in PyEphem")
parser.add_argument("--doPlot", "-dp", dest="plot", action="store_true", help="Make plots for specified field")
parser.add_argument("--simulate", "-s", dest="sim", action="store_true", help="Simulate only -- make no modifications to database")
parser.add_argument("--parangstep", "-pas", type=float, dest="stepsize", default=1., help="Parallactic angle correction step size in minutes")
parser.add_argument("--chunksize", "-cs", type=int, dest="chunksize", default=1000, help="Chunk size in rows")
parser.add_argument("--datadiscriptor", "-dd", type=int, dest="ddid", default=0, help="Select data descriptor (SPW)")
parser.add_argument("--applyantidiag", "-ad", dest="flipfeeds", action="store_true", help="Apply anti-diagonal matrix -- flips the visibilty hands")


args = parser.parse_args()

if args.plot:
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    log.info("Enabling plotting")

meerkat = ephem.Observer()
meerkat.lat = "-30:42:47.41"
meerkat.long = "21:26:38.0"
meerkat.elevation = 1054
meerkat.epoch = ephem.J2000

with tbl(args.ms, ack=False) as t:
    with taql("select * from $t where FIELD_ID=={}".format(args.field)) as tt:
        def __to_datetimestr(t):
            dt = datetime.datetime.utcfromtimestamp(quantity("{}s".format(t)).to_unix_time())
            return dt.strftime("%Y/%m/%d %H:%M:%S")
        dbtime = tt.getcol("TIME_CENTROID")
        start_time_Z = __to_datetimestr(dbtime.min())
        end_time_Z = __to_datetimestr(dbtime.max())
        log.info("Computing parallacting angles between '{}' and '{}' UTC".format(
                 start_time_Z, end_time_Z))

meerkat.date = start_time_Z
st = meerkat.date
meerkat.date = end_time_Z
et = meerkat.date
TO_SEC = 3600*24.0
nstep = int(np.round((float(et)*TO_SEC - float(st)*TO_SEC) / (args.stepsize*60.)))
log.info("Computing PA in {} steps of {} mins each".format(nstep, args.stepsize))
timepa = time = np.linspace(st,et,nstep)
timepadt = list(map(lambda x: ephem.Date(x).datetime(), time))

if args.ephem: 
    with tbl(args.ms+"::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
    fieldEphem = getattr(ephem, args.ephem, None)()
    if not fieldEphem:
        raise RuntimeError("Body {} not defined by PyEphem".format(args.ephem))
    log.info("Overriding stored ephemeris in database '{}' field '{}' by special PyEphem body '{}'".format(
        args.ms, fieldnames[args.field], args.ephem))
else:
    with tbl(args.ms+"::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
        pos = t.getcol("PHASE_DIR")
    skypos = SkyCoord(pos[args.field][0,0]*units.rad, pos[args.field][0,1]*units.rad, frame="fk5")
    rahms = "{0:.0f}:{1:.0f}:{2:.5f}".format(*skypos.ra.hms)
    decdms = "{0:.0f}:{1:.0f}:{2:.5f}".format(skypos.dec.dms[0], abs(skypos.dec.dms[1]), abs(skypos.dec.dms[2]))
    fieldEphem = ephem.readdb(",f|J,{},{},0.0".format(rahms, decdms))
    log.info("Using coordinates of field '{}' for body: J2000, {}, {}".format(fieldnames[args.field],
                                                                              np.rad2deg(pos[args.field][0,0]),
                                                                              np.rad2deg(pos[args.field][0,1])))

az = np.zeros(nstep, dtype=np.float32)
el = az.copy()
ra = az.copy()
dec = az.copy()
pa = az.copy()

for ti, t in enumerate(time):
    meerkat.date = t
    fieldEphem.compute(meerkat)
    az[ti] = fieldEphem.az
    el[ti] = fieldEphem.alt
    ra[ti] = fieldEphem.ra
    dec[ti] = fieldEphem.dec
    pa[ti] = fieldEphem.parallactic_angle()

if args.plot:
    for axl, axd in zip(["Az", "El", "RA", "DEC", "ParAng"],
                        [az, el, ra, dec, pa]):
        hfmt = mdates.DateFormatter('%H:%M')
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Time UTC")
        ax.set_ylabel("{} [deg]".format(axl))
        ax.plot(timepadt, np.rad2deg(axd))
        ax.xaxis.set_major_formatter(hfmt)
        ax.grid(True)
        plt.show()

with tbl(args.ms+"::ANTENNA", ack=False) as t:
    anames = t.getcol("NAME")

with tbl(args.ms+"::FEED", ack=False) as t:
    receptor_aid = t.getcol("ANTENNA_ID")
    if len(receptor_aid) != len(anames):
        raise RuntimeError("Receptor angles not all filed for the antennas in the ::FEED keyword table")
    receptor_angles = dict(zip(receptor_aid, t.getcol("RECEPTOR_ANGLE")[:,0]))
    log.info("Applying the following feed angle offsets to parallactic angles:")
    for ai, an in enumerate(anames):
        log.info("\t {0:s}: {1:.3f} degrees".format(an, np.rad2deg(receptor_angles.get(ai, 0.0))))

with tbl(args.ms+"::POLARIZATION", ack=False) as t:
    poltype = t.getcol("CORR_TYPE")
    # must be linear
    for p in poltype:
        if any(p - np.array([9,10,11,12]) != 0):
            raise RuntimeError("Must be full correlation linear system being corrected")

with tbl(args.ms+"::DATA_DESCRIPTION", ack=False) as t:
    if args.ddid < 0 or args.ddid >= t.nrows():
        raise RuntimeError("Invalid DDID selected")
    spwsel = t.getcol("SPECTRAL_WINDOW_ID")[args.ddid]

with tbl(args.ms+"::SPECTRAL_WINDOW", ack=False) as t:
    chan_freqs = t.getcol("CHAN_FREQ")[spwsel]
    chan_width = t.getcol("CHAN_WIDTH")[spwsel]
    nchan = chan_freqs.size
    log.info("Will apply to SPW {0:d} ({3:d} channels): {1:.2f} to {2:.2f} MHz".format(
        spwsel, chan_freqs.min()*1e-6, chan_freqs.max()*1e-6, nchan))

if args.flipfeeds:
    log.info("Will flip the visibility hands per user request")

if not args.sim:
    log.info("Applying P Jones")
    timepaunix = np.array(list(map(lambda x: x.replace(tzinfo=pytz.UTC).timestamp(), timepadt)))
    raarr = np.empty(len(anames), dtype=int)
    for aid in range(len(anames)):
        raarr[aid] = receptor_angles[aid]
    nrowsput = 0
    with tbl(args.ms, ack=False, readonly=False) as t:
        with taql("select * from $t where FIELD_ID=={}".format(args.field)) as tt:
            nrow = tt.nrows()
            nchunk = nrow // args.chunksize + int(nrow % args.chunksize > 0)
            for ci in range(nchunk):
                cl = ci * args.chunksize
                crow = min(nrow - ci * args.chunksize, args.chunksize)
                data = tt.getcol("DATA", startrow=cl, nrow=crow)
                if data.shape[2] != 4:
                    raise RuntimeError("Data must be full correlation")
                data = data.reshape(crow, nchan, 2, 2)

                def __casa_to_unixtime(t):
                    dt = quantity("{}s".format(t)).to_unix_time()
                    return dt
                timemsunix = np.array(list(map(__casa_to_unixtime,
                                           tt.getcol("TIME_CENTROID", startrow=cl, nrow=crow))))
                a1 = tt.getcol("ANTENNA1", startrow=cl, nrow=crow)
                a2 = tt.getcol("ANTENNA2", startrow=cl, nrow=crow)

                # nearest neighbour interp to computed ParAng
                pamap = np.array(list(map(lambda x: np.argmin(np.abs(x - timepaunix)), timemsunix)))

                # apply receptor angles and get a PA to apply per row
                # assume same PA for all antennas, different F Jones per antenna possibly
                paA1 = pa[pamap] + raarr[a1]
                paA2 = pa[pamap] + raarr[a2]

                def give_lin_Rmat(paA, nchan, conjugate=False):
                    N = paA.shape[0] # nrow
                    c = np.cos(paA).repeat(nchan)
                    s = np.sin(paA).repeat(nchan)
                    if conjugate:
                        return np.array([c,s,-s,c]).T.reshape(N, nchan, 2, 2)
                    else:
                        return np.array([c,-s,s,c]).T.reshape(N, nchan, 2, 2)

                # need to apply anti-diagonal
                if args.flipfeeds:
                    FVmat = np.array([np.zeros(nchan*crow),
                                      np.ones(nchan*crow),
                                      np.ones(nchan*crow),
                                      np.zeros(nchan*crow)]).T.reshape(crow, nchan, 2, 2)
                else: # ignore step
                    FVmat = np.array([np.ones(nchan*crow),
                                      np.zeros(nchan*crow),
                                      np.zeros(nchan*crow),
                                      np.ones(nchan*crow)]).T.reshape(crow, nchan, 2, 2)

                PA1 = give_lin_Rmat(paA1, nchan=nchan)
                PA2 = give_lin_Rmat(paA2, nchan=nchan, conjugate=True)
                JA1 = np.matmul(PA1, FVmat)
                JA2 = np.matmul(FVmat, PA2)
                corr_data = np.matmul(JA1, np.matmul(data, JA2)).reshape(crow, nchan, 4)
                tt.putcol("CORRECTED_DATA", corr_data, startrow=cl, nrow=crow)
                log.info("\tCorrected chunk {}/{}".format(ci+1, nchunk))
                nrowsput += crow
        assert nrow == nrowsput
else:
    log.info("Simulating correction only -- no changes applied to data")
