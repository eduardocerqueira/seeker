#date: 2025-02-11T17:06:55Z
#url: https://api.github.com/gists/a66c872fc8ea2b4b0d9ff5cf1667ac0c
#owner: https://api.github.com/users/roland3564

# pip install vedastro
# pip install pandas
# pip install geopy
# pip install pytz
# pip install timezonefinder
# pip install pprint

from pprint import pprint
import pandas as pd

# import pytz
# import math

# from timezonefinder import TimezoneFinder
# from datetime import datetime
# from vedastro import *
# from geopy.geocoders import *

# def get_timezone_and_offset(latitude, longitude, time_str):
#     tf = TimezoneFinder()
#
#     timezone_name = tf.timezone_at(lng=longitude, lat=latitude)
#
#     if timezone_name is None:
#         return None, None
#
#     naive_time = datetime.strptime(time_str, "%H:%M")
#
#     tz = pytz.timezone(timezone_name)
#     localized_time = tz.localize(naive_time)
#
#     utc_offset = localized_time.utcoffset().total_seconds() / 3600
#     return timezone_name, utc_offset

# def get_coordinates(city, country, axis):
#     geolocator = Nominatim(user_agent="geo_locator")
#     location = geolocator.geocode(f"{city}, {country}")
#
#     if location:
#         if   str.lower(axis)=='y':
#             return location.latitude
#         elif str.lower(axis)=='x':
#             return location.longitude
#         else:
#             return None
#     else:
#         return None

def main():
    anagrafe = {
            'Nome':     ['Roland', 'Anna'],
            'Data':     ['13/02/2004', '08/11/1977'],
            'Ora':      ['22:36', '09:00'],
            'Country':  ['Russia', 'Russia'],
            'City':     ['Italy', 'Ulyanovsk']}

    df = pd.DataFrame(anagrafe)

    # lat = float(get_coordinates(df.loc[0, 'Country'], df.loc[0, 'City'], 'y'))
    # lon = float(get_coordinates(df.loc[0, 'Country'], df.loc[0, 'City'], 'x'))
    # lat = round(lat, 2)
    # lon = round(lon, 2)

    # date=str(df.loc[0, 'Data'])
    # time=str(df.loc[0, 'Ora'])
    # country= str(df.loc[0, 'Country'])
    # city= str(df.loc[0, 'City'])
    # location= country+", "+city
    # timezone, offset = get_timezone_and_offset(lat, lon, time)
    # offset= math.ceil(offset)

    # if offset >= 0:
    #     if offset >= 10:
    #         tz="+"+str(offset)+":00"
    #     else:
    #         tz="+0"+str(offset)+":00"
    # else:
    #     if offset <= 10:
    #         tz="-"+str(offset)+":00"
    #     else:
    #         tz="-0"+str(offset)+":00"

    # geolocation= GeoLocation(location, lon, lat)
    # timestring=time+" "+date+" "+tz
    # birth_time= Time(timestring, geolocation)

    # print(location)
    # print(date," and ", time)
    # print(lat, " and ", lon)
    # print("timezone "+tz+"\n")

    # PLANETS
    # allPlanetDataList = Calculate.AllPlanetData(PlanetName.Sun, birth_time)
    # HOUSES
    # allHouseDataList = Calculate.AllHouseData(HouseName.House1, birth_time)
    # ZODIAC SIGNS
    # allZodiacDataList = Calculate.AllZodiacSignData(ZodiacName.Gemini, birth_time)

    # df1a = pd.json_normalize(allPlanetDataList, sep='_')
    # df1b = pd.json_normalize(allHouseDataList, sep='_')
    # df1c = pd.json_normalize(allZodiacDataList, sep='_')

    # # Convert first records to dictionaries
    # df1af = df1a.to_dict(orient='records')[0]
    # df1bf = df1b.to_dict(orient='records')[0]
    # df1cf = df1c.to_dict(orient='records')[0]

    # # Merge dictionaries
    # merged_data = {**df1af, **df1bf, **df1cf}  # Avoids modifying original dict

    # # Convert merged dict to DataFrame properly
    # df1 = pd.DataFrame([merged_data])

    df=df
    pprint(df)
    print("\n")
    pprint(df.head(1))
    print("\n")
    df.info()
    print("\n")
    df.describe()
    print("\n")
    df.drop("City", axis=1, inplace= True)
    pprint(df)
    df.drop(1, axis=0, inplace=True)
    print("\n")
    pprint(df)
    df.replace("Russia","Giappone", inplace=True)
    print("\n")
    pprint(df)
    df.replace("22:36",None, inplace=True)
    print("\n")
    pprint(df)
    df.fillna("00:00", inplace=True)
    print("\n")
    pprint(df)
    df.rename(columns={"Country":"LuogoDiResidenza"}, inplace=True)
    print("\n")
    pprint(df)

    anagrafe = {
        'Nome': ['Roland', 'Michiro', 'Yumi', 'Izumi'],
        'Age': [21, 77, 24, 25],
        'Height': [176, 170, 172, 177],
        'City': ['Tokyo', 'Osaka', 'Yokohama', 'Tokyo']}

    df=pd.DataFrame(anagrafe)
    print("\n")
    pprint(df)
    print("\n")
    pprint(df['Age'].mean())
    pprint(df['Height'].sum())
    pprint(df['Height'].max())
    pprint(df.groupby('City')['Height'].max())

    print("\n")

    pprint(df.groupby('City').agg({'Age': ['sum', 'mean']}))

    print("\n")

    df1 = pd.DataFrame({
        'col1': [1,2,3],
        'col2': ['a','b','c'],
        'col3': [4,5,6]
    })

    select = df1[['col1','col2']]
    print("\n")
    pprint(select)

    print("\n")

    filtered= df1[df1['col3']>5]
    pprint(filtered)

    dfb = pd.DataFrame({
        'id': [1,2,4],
        'nome': ["Alpha","Beta","Omega"]
    })

    print("\n")

    dfa = pd.DataFrame({
        'id': [1, 2, 3],
        'nome': ["Alpha", "Bravo", "Charlie"]
    })

    joined= pd.merge(dfa,dfb,on='id',how='inner')
    pprint(joined)

    print("\n")

    joined = pd.merge(dfa, dfb, on='id', how='right')
    pprint(joined)

    dfc=pd.DataFrame({
        'prodotto': ['A','B','A'],
        'prezzo': [4,2,7]
    })

    grouped=dfc.groupby('prodotto').mean()
    print("\n")
    pprint(grouped)

if __name__ == "__main__":
    main()