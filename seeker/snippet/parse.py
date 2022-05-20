#date: 2022-05-20T17:17:14Z
#url: https://api.github.com/gists/1d22c4e5bf906565d4fb986077a58de7
#owner: https://api.github.com/users/ianyoung

from utils import title, images

def parse_properties(agent, raw, customer_id, access_token):
    """
    Parse the list of properties and and convert to a standard output.
    Return a formatted list of properties.
    """
    properties = []

    for item in raw['_embedded']:

        image_list, plans_list = images.get_images(item, customer_id, access_token)

        property = {
            'ref': item['id'],
            'title': title.get_title(item),
            "images": image_list,
            "plans": plans_list
        }
        properties.append(property)

    return properties