#date: 2022-01-05T17:19:28Z
#url: https://api.github.com/gists/0f90760f16470cd1b3ec5d9106ecc160
#owner: https://api.github.com/users/maikhoepfel

# Excerpt for https://github.com/etsy/open-api/issues/27
# Untested, unproven, don't use this in production!

def _make_request(method, url, data=None):
    with open("etsy_token.json", "r") as f:
        token = json.loads(f.read())
    oauth = OAuth2Session(client_id, token=token)
    oauth.refresh_token(
        "https://api.etsy.com/v3/public/oauth/token", client_id=client_id
    )
    response = oauth.request(method, url, json=data, headers={"x-api-key": client_id})
    response.raise_for_status()
    return response.json()


def set_price(listing_id, sku, price):
    """
    Sets price on a single product or variant

    Rather complex because Etsy makes us submit all variants and their properties
    to change a single price. We have to inspect the given listing to rebuild
    dicts that look almost like what is given, but not quite.
    
    Does not support property combinations, and does not support multiple 
    offerings per product (not sure if that's a thing)

    https://developers.etsy.com/documentation/reference#operation/updateListingInventory
    """
    listing = get_inventory(listing_id)
    assert len(listing["price_on_property"]) <= 1
    assert len(listing["quantity_on_property"]) <= 1
    assert len(listing["sku_on_property"]) <= 1

    matching_products = [
        product for product in listing["products"] if product["sku"] == sku
    ]
    assert len(matching_products) == 1

    data = {
        "price_on_property": listing["price_on_property"],
        "quantity_on_property": listing["quantity_on_property"],
        "sku_on_property": listing["sku_on_property"],
        "products": [],
    }
    for product in listing["products"]:
        assert len(product["offerings"]) == 1
        offer = product["offerings"][0]
        if product["sku"] == sku:
            price_to_set = price
        else:
            price_to_set = float(offer["price"]["amount"]) / offer["price"]["divisor"]
        product_data = {
            "sku": product["sku"],
            "offerings": [
                {
                    "price": price_to_set,
                    "quantity": offer["quantity"],
                    "is_enabled": offer["is_enabled"],
                }
            ],
        }
        values = product.get("property_values")
        if values:
            assert len(values) == 1
            value = values[0]
            value_data = {
                "property_id": value["property_id"],
                "value_ids": value["value_ids"],
                "property_name": value["property_name"],
                "values": value["values"],
            }
            if value["scale_id"] and value["scale_id"] != "None":
                value_data["scale_id"] = value["scale_id"]
            product_data["property_values"] = [value_data]

        data["products"].append(product_data)

    return _make_request(
        "PUT",
        f"https://openapi.etsy.com/v3/application/listings/{listing_id}/inventory",
        data=data,
    )