#date: 2023-07-06T17:04:28Z
#url: https://api.github.com/gists/a231545a066e0fe5fd30dcccec4c0e86
#owner: https://api.github.com/users/peterryder3


from arcgis.gis import GIS
from datetime import datetime, timedelta
gis = GIS("home")
DELETE_DAYS = 30
SERVICE_MAP_ID = '3eff8280a0fb448cad62dbe9406516f9'
# Calculate the date 3 months ago
WHERE_CLAUSE = "Processed = 'Yes'"
three_months_ago = datetime.now() - timedelta(days=DELETE_DAYS)

# ADD TO THIS LIST THE FEATURE SERVICE KEYS (THESE MUST MATCH THE KEYS IN THE SERVICE ID TABLE )
to_delete_keys= ['ProdClubSingleUploader',
                 'ProdClubBulkUploader',
                 'ProdActivityBulkUploader',
                 'ProdActivitySingleUploader']

delete_feats = []
for key in to_delete_keys:
    #grab all source feature layer
    id1  = gis.content.search(SERVICE_MAP_ID, max_items=-1)[0].tables[0].query(f"Service_Name ='{key}'").features[0].attributes.get("ID")
    layer = gis.content.search(id1, item_type='Feature Service')[0].layers[0]
    feat = layer.query(where = WHERE_CLAUSE)      
    for feat in feat:
        # Convert the timestamp to a datetime object
        ts1 = int(feat.attributes['CreationDate'])/1000
        dobj = datetime.fromtimestamp(ts1)
        if dobj < three_months_ago:
            delete_feats.append(feat)
            
    if delete_feats:
        layer.delete_features(deletes=delete_feats)
        delete_feats = []