#date: 2022-06-30T21:12:56Z
#url: https://api.github.com/gists/2757e58081a1ae377a9c0feed875976b
#owner: https://api.github.com/users/jasonsnell

#! /usr/bin/env python3

import requests
import time
import jwt
import json
import requests
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

theDayCondition = defaultdict(lambda: -99999)
hitemp = defaultdict(lambda: -99999)
lotemp = defaultdict(lambda: 99999)
datelist = defaultdict(lambda: 99999)
diff = {}
theDates = [1, 2, 3, 4, 5, 6, 7, 8]

# all these variables below need to be customized

lat = "your-latitude"
lon = "your-longitude"
chartSavePath = '/Users/your-user-name/Desktop/'
team_id = "your_team_id"
service_id = "com.yourdomain.weatherproject"
key_id = "your_key_id"
private_key = "-----YOUR PRIVATE KEY-----"

def symbolFromConditions(theCondition, theRequest):

    conditionSymbol = ""
    if theCondition == "Clear":
        symbolName = "sun.max"
        theSFSymbol = "􀆭"
    elif theCondition == "MostlyClear":
        symbolName = "sun.min"
        theSFSymbol = "􀆫"
    elif theCondition == "PartlyCloudy":
        symbolName = "cloud.sun"
        theSFSymbol = "􀇔"
    elif theCondition == "MostlyCloudy":
        symbolName = "cloud"
        theSFSymbol = "􀇂"
    elif theCondition == "Cloudy":
        symbolName = "cloud"
        theSFSymbol = "􀇂"
    elif theCondition == "Hazy":
        symbolName = "sun.haze"
        theSFSymbol = "􀆷"
    elif theCondition == "ScatteredThunderstorms":
        symbolName = "cloud.sun.bolt"
        theSFSymbol = "􀇘"
    elif theCondition == "Drizzle":
        symbolName = "cloud.drizzle"
        theSFSymbol = "􀇄"
    elif theCondition == "rain":
        symbolName = "cloud.rain"
        theSFSymbol = "􀇆"
    elif theCondition == "HeavyRain":
        symbolName = "cloud.heavyrain"
        theSFSymbol = "􀇈"
    else:
        symbolName = theCondition
        theSFSymbol = theCondition
    if theRequest == "sf":
        return theSFSymbol
    else:
        return (f':{symbolName}: ')

expiry = (int(time.time()) + 60)

encoded_jwt = jwt.encode({"iss": team_id, "iat": int(time.time()), "exp": expiry, "sub": service_id}, private_key, algorithm="ES256", headers={"kid": key_id, "id": (team_id + '.' + service_id)})

dailyurl = f"https://weatherkit.apple.com/api/v1/weather/en/{lat}/{lon}?dataSets=forecastDaily&timezone=America/Los_Angeles"

d = requests.get(dailyurl, headers={"Authorization":("Bearer " + encoded_jwt)})

if len(d.content) < 1:
    raise Exception("File not retrieved")

# dailyForecast is the variable containing the entire forecast payload
dailyForecast = json.loads(d.content)

todayHiC = dailyForecast['forecastDaily']['days'][0]['temperatureMax']
todayLoC = dailyForecast['forecastDaily']['days'][0]['temperatureMin']
todayHiF = int((todayHiC * 9/5) + 32)
todayLoF = int((todayLoC * 9/5) + 32)
todayCondition = dailyForecast['forecastDaily']['days'][0]['restOfDayForecast']['conditionCode']
todayPrecip = dailyForecast['forecastDaily']['days'][0]['restOfDayForecast']['precipitationType']

if todayPrecip != "clear":
    conditionSymbol = symbolFromConditions(todayPrecip, '')
else:
    conditionSymbol = symbolFromConditions(todayCondition, '')

print (f'{conditionSymbol} {todayHiF}°/{todayLoF}°')

for dayItem in range(1,9):
    thisHiC = dailyForecast['forecastDaily']['days'][dayItem]['temperatureMax']
    hitemp[dayItem] = int((thisHiC * 9/5) + 32)
    theDay = dailyForecast['forecastDaily']['days'][dayItem]['forecastStart']
    theDayProper = datetime.strptime(theDay, "%Y-%m-%dT%H:%M:%SZ")
    datelist[dayItem] = datetime.strftime(theDayProper, "%a")
    theDayCondition[dayItem] = dailyForecast['forecastDaily']['days'][dayItem]['conditionCode']
    thatPrecip = dailyForecast['forecastDaily']['days'][dayItem]['precipitationType']
    # print(thatPrecip)
    if thatPrecip != "clear":
        theDayCondition[dayItem] = symbolFromConditions(thatPrecip, 'sf')
    else:
        theDayCondition[dayItem] = symbolFromConditions(theDayCondition[dayItem], 'sf')

theHighs = list(hitemp.values())
theDateList = list(datelist.values())
theConditionsList = list(theDayCondition.values())

himin = min(theHighs) - 8

# Graphing

fig, ax = plt.subplots()

font = {'fontname': 'SF Pro'}
ax.bar(theDates, theHighs, color="#333", alpha=0)
ax.bar_label(ax.containers[0], color="black", fontweight='bold')
ax.set_aspect(aspect=0.12)
ax.set_yticks([])
for i in range(0,8):
    plt.text((i+1),(himin + 2),str(theConditionsList[i]), color="black", horizontalalignment='center', size='x-large', fontweight='bold', **font)
plt.box(False)
plt.ylim(ymin=himin)
plt.xticks(theDates,  theDateList, color="black", fontweight='bold')
plt.savefig(f'{chartSavePath}forecast.png', dpi=300, bbox_inches='tight',
            pad_inches=0.05)
