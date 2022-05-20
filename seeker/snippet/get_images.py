#date: 2022-05-20T17:17:14Z
#url: https://api.github.com/gists/1d22c4e5bf906565d4fb986077a58de7
#owner: https://api.github.com/users/ianyoung

import os
import requests
from operator import itemgetter


def get_images(item, customer_id, access_token):
    """
    Returns a list of images (dictionaries) including URL & description.

    Parameters:
        item (dict): Property
        customer_id (str): A 3 char client identifier provided by the agent.
        access_token (str): Bearer token for authenticating the request.

    Returns:
        images_cleaned (List[dict]): List of formatted image items.
        floorplans_cleaned (List[dict]): List of formatted floorplan items.
    """
    images_url = os.getenv('IMAGES_URL')
    property_id = item['id']

    headers = {
        'accept': 'application/json',
        'authorization': 'Bearer ' + access_token
    }

    params = {
        'propertyId': property_id,
        'pageSize': 50, # Limits images to 50 (max 100 before needing to paginate) 
        'pageNumber': 1
    }

    # Make the request
    response = requests.get(images_url, headers=headers, params=params)
    raw_json = response.json()
    
    images_unordered = []

    for img in raw_json['_embedded']:

        # Build a dictionary of image items with fallback values for keys.
        image = {
            'url': img['url'] or None,
            'description': img['caption'] or None,
            'type': img['type'] or None,  # Used for identifying floorplans.
            'order': img['order'] or None # Used for setting image order.
        }

        # Build a list of unordered images
        images_unordered.append(image)

        # Reorder the images according to the `order` key.
        images_ordered = sorted(images_unordered, key=itemgetter('order')) 
        
        # Extract the floorplans into their own list
        floorplans = [i for i in images_ordered if i['type'] == 'floorPlan']

        # Remove the floorplans from the images list
        images = [i for i in images_ordered if i['type'] != 'floorPlan']

        # Remove `order` and `type` keys from dictionaries in the images list.
        images_cleaned = [{k:v for k,v in i.items() if k != 'order' and k != 'type'} for i in images]

        # Remove `order` and `type` keys from dictionaries in the floorplans list.
        floorplans_cleaned = [{k:v for k,v in i.items() if k != 'order' and k != 'type'} for i in floorplans]

    return images_cleaned, floorplans_cleaned
