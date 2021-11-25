#date: 2021-11-25T17:02:06Z
#url: https://api.github.com/gists/7ea4b2998ee80f37dd332a9ab3d381da
#owner: https://api.github.com/users/denis-bz

#!/usr/bin/env python3
"""Wind power = power curve * wind speed histogram """

from io import StringIO
import numpy as np

print( 80 * "▄" )
print( __doc__ )

#...............................................................................
powercurve = """\
0 1 2 3   4   5   6    7    8    9   10   11   12   13   14   15   16   17   18   19   20  # wind speed m/s
0 0 0 0 225 494 895 1455 2200 3156 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000  # kilowatts
"""
percents = """\
0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20  # wind speed m/s
0  3  6  8 10 11 11 10  9  8  6  5  4  3  2  1  1  1  0  0  0  # %
"""

print( "In: a wind power curve, kilowatts at wind speed 0 1 2 ... m/s: \n"
            + powercurve )
print( "In: percent wind at 0 1 2 ... m/s (a histogram, here Weibull av=7 m/s, k=2 ): \n"
            + percents )
print( """Out: total kilowatts = sum of (KW * % at each windspeed)
    (KW at 0 1 2 ... m/s) *
    (%  at 0 1 2 ... m/s)\
""" )

x, kilowatts = np.loadtxt( StringIO( powercurve ))  # 2 × n ndarray
x, percents = np.loadtxt( StringIO( percents ))

#...............................................................................
kw_at_each_windspeed = kilowatts * percents / 100
print( "\nKilowatts at each windspeed:" )
print( np.vstack( (x, kw_at_each_windspeed )) .astype( int ))

totalkw = kw_at_each_windspeed.sum()
percentmax = totalkw / kilowatts.max() * 100

print( "\nTotal: %.0f KW = %.0f %% of max" % (  # 1691 KW = 42 % of max
            totalkw, percentmax ))

