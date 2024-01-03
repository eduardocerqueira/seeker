#date: 2024-01-03T17:01:29Z
#url: https://api.github.com/gists/6cb551869fb719715cbf5ebc91aebeab
#owner: https://api.github.com/users/ryleysevier

import os, logging

from hubspot import HubSpot
from hubspot.crm.line_items import BatchReadInputSimplePublicObjectId
from hubspot.crm.associations import BatchInputPublicObjectId


def main():
    # Initialize the API client
    api_client = "**********"='pat-na1-')

    # get your deals
    deals = api_client.crm.deals.get_all()

    # This is the batch element for associations so you're minimizing calls
    deal_ids = BatchInputPublicObjectId([{'id':str(deal.id)} for deal in deals])
    # The oobjects types are listed here: https://developers.hubspot.com/docs/api/crm/understanding-the-crm#object-type-id
    line_items = api_client.crm.associations.batch_api.read(from_object_type='DEAL', to_object_type='LINE_ITEM', batch_input_public_object_id=deal_ids)

    # quick out to simplify down to dealkey:lineitemkey_array
    deals_dict = {}
    line_item_raw = []
    for d in line_items.results:
        deals_dict[d._from.id] = [l.id for l in d.to]
        line_item_raw.extend([{"id": i.id} for i in d.to])

    # use batch to populate line items data
    hs_line_item_props = ['price', 'quantity', 'name']
    batch_read_input_simple_public_object_id = BatchReadInputSimplePublicObjectId(
        inputs=line_item_raw, 
        properties=hs_line_item_props)
    line_items = api_client.crm.line_items.batch_api.read(batch_read_input_simple_public_object_id=batch_read_input_simple_public_object_id)

    line_item_objs = {}
    for r in line_items.results:
        line_item_objs[r.id] = r

    out = {}
    # you might not want to simplify objects, you can adjust this to return the full object
    for k,v in deals_dict.items():
        out[k] = [line_item_objs[i].properties for i in v]

    logging.info(out)


if __name__ == "__main__":
    main()  main()