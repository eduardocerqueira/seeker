#date: 2022-01-26T17:04:18Z
#url: https://api.github.com/gists/414247bab678ad193ed174fa9218be60
#owner: https://api.github.com/users/kc9ryt

# I use a Python script to pull current weather conditions from the NOAA web service API. The NOAA web
# service does not return a windchill value for all locations, but given temperature, relative humidity,
# and wind speed you can calculate a “feels like” temperature as follows.

# This code assumes units of Fahrenheit, MPH, and Relative Humidity by percentage.  In this example, a
# temperature of 35F, wind speed of 10mph, and relative humidity of 72% yields a "feels like" value of 27.4F
 
import math
 
vTemperature = float(35)
vWindSpeed = float(10)
vRelativeHumidity = float(72)
 
# Try Wind Chill first
if vTemperature <= 50 and vWindSpeed >= 3:
  vFeelsLike = 35.74 + (0.6215*vTemperature) - 35.75*(vWindSpeed**0.16) + ((0.4275*vTemperature)*(vWindSpeed**0.16))
else:
  vFeelsLike = vTemperature
 
# Replace it with the Heat Index, if necessary
if vFeelsLike == vTemperature and vTemperature >= 80:
  vFeelsLike = 0.5 * (vTemperature + 61.0 + ((vTemperature-68.0)*1.2) + (vRelativeHumidity*0.094))
 
  if vFeelsLike >= 80:
    vFeelsLike = -42.379 + 2.04901523*vTemperature + 10.14333127*vRelativeHumidity - .22475541*vTemperature*vRelativeHumidity - .00683783*vTemperature*vTemperature - .05481717*vRelativeHumidity*vRelativeHumidity + .00122874*vTemperature*vTemperature*vRelativeHumidity + .00085282*vTemperature*vRelativeHumidity*vRelativeHumidity - .00000199*vTemperature*vTemperature*vRelativeHumidity*vRelativeHumidity
    if vRelativeHumidity < 13 and vTemperature >= 80 and vTemperature <= 112:
      vFeelsLike = vFeelsLike - ((13-vRelativeHumidity)/4)*math.sqrt((17-math.fabs(vTemperature-95.))/17)
      if vRelativeHumidity > 85 and vTemperature >= 80 and vTemperature <= 87:
        vFeelsLike = vFeelsLike + ((vRelativeHumidity-85)/10) * ((87-vTemperature)/5)				
 
print "Feels like: " + '%0.1f' % (vFeelsLike) + "F"
# Feels like: 27.4F