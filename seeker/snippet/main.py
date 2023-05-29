#date: 2023-05-29T16:47:15Z
#url: https://api.github.com/gists/e0027e05bd6a646d3235b44f102c905d
#owner: https://api.github.com/users/Danielvasquezcf

import unittest
import requests


class EasyBrokerAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def get_properties(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        url = f"{self.base_url}/properties"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            properties = response.json()
            return properties
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

    def print_property_titles(self):
        properties = self.get_properties()
        if properties:
            for property in properties:
                title = property.get("title")
                print(title)


# Ejemplo de uso
api_key = "l7u502p8v46ba3ppgvj5y2aad50lb9"
base_url = "https://api.easybroker.com/v1/properties"

easy_broker_api = EasyBrokerAPI(base_url, api_key)
easy_broker_api.print_property_titles()

###################################




class TestEasyBrokerAPI(unittest.TestCase):
    def test_print_property_titles(self):
        api = EasyBrokerAPI("https://api.easybroker.com/v1/properties", "l7u502p8v46ba3ppgvj5y2aad50lb9")
        api.print_property_titles()

if __name__ == '__main__':
    unittest.main()