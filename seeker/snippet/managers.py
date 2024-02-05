#date: 2024-02-05T16:56:31Z
#url: https://api.github.com/gists/9fe42986d639e5c33258599b748d51a7
#owner: https://api.github.com/users/rtu-dataframe

from django.db import models, connection

"""
Haversine formula in Django using Postgres SQL
Queries a model for all rows that are within a certain radius (given in meters) of a central point.

The 'location_model' placeholder should be raplaced with the name of the table (full name) that includes a latitude and longitude column.

The latitude and longitude columns should be decimal fields.

Returns a list of row ids.
"""
class LocationManager(models.Manager):
    def in_range(self, latitude, longitude, radius, results=100):
        unit = 6371 # Distance unit (kms)
        radius = float(radius) / 1000.0 # Distance radius convert m to km
        latitude = float(latitude) # Central point latitude
        longitude = float(longitude) # Central point longitude

        sql = """SELECT id FROM
                    (SELECT id, latitude, longitude, ({unit} * acos(CAST((cos(radians({latitude})) * cos(radians(latitude)) *
                                                     cos(radians(longitude) - radians({longitude})) +
                                                     sin(radians({latitude})) * sin(radians(latitude))) AS DECIMAL)))
                     AS distance
                     FROM location_model) AS distances
                 WHERE distance < {radius}
                 ORDER BY distance
                 OFFSET 0
                 LIMIT {results};""".format(unit=unit, latitude=latitude, longitude=longitude, radius=radius, results=results)

        cursor = connection.cursor()
        cursor.execute(sql)
        ids = [row[0] for row in cursor.fetchall()]

        return ids
