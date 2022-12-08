#date: 2022-12-08T17:10:09Z
#url: https://api.github.com/gists/2781f29381dbe8e4021ba4d0fa66b23e
#owner: https://api.github.com/users/christopherDT

import pandas as pd
from onboard.client.models import PointSelector

# buildings in the portfolio
building_df = pd.json_normalize(client.get_all_buildings())

# generate a query for our two points
query = PointSelector()
query.point_ids = [518876, 518877] # these are points in the Laboratory
selection = client.select_points(query)

# query and merge metadata from points, equipment, and buildings:

# point data
sensor_metadata = client.get_points_by_ids(selection['points'])
sensor_metadata_df = pd.DataFrame(sensor_metadata)

# equipment data
equip_metadata = client.get_equipment_by_ids(selection['equipment'])
equip_metadata_df = pd.DataFrame(equip_metadata)

# join sensor and equip and building meta
df = pd.merge(
    sensor_metadata_df,
    equip_metadata_df.drop(
        columns=['building_id']).rename(
        columns={"id": "equip_id", "equip_id": "equip_name"}),
    left_on='equip_id', right_on='equip_id', how='left').merge(
    building_df[['id', 'name', 'address']].rename(
        columns={"id": "building_id", "name": "building_name"}),
    left_on='building_id', right_on='building_id', how='left'
)

display_cols = ['id','building_name','equip_name', 'name', 'type',]
bldg_df = df[display_cols + ['value', 'units']]
bldg_df
