#date: 2021-12-28T16:48:35Z
#url: https://api.github.com/gists/7d5b2e3a6e6e45f399e64606e5d21489
#owner: https://api.github.com/users/nicmedina

"""
Created on Tue Dec 28 10:07:43 2021

@author: Nicolas
"""

import unittest    #import libraries
from internship1 import getProperties

class TestCaseIntern(unittest.TestCase):    #start testclass
    def test_intern(self):
        url = 'https://api.stagingeb.com/v1/properties'  #provide the URL for easyBroker API
        headers = {'X-Authorization': 'l7u502p8v46ba3ppgvj5y2aad50lb9'}  #provide access key to EB API
        args = {'updated_after': '2020-03-01T23:26:53.402Z',    #sete the filters for the search
                'updated_before': '2025-03-01T23:26:53.402Z',
                'operation_type': 'sale',
                'min_price': 500000,
                'max_price': 3000000,
                'min_bedrooms': 1,
                'min_bathrooms': 1,
                'min_parking_spaces': 1,
                'min_construction_size': 100,
                'max_construction_size': 1000,
                'min_lot_size': 100,
                'max_lot_size': 1000}
        expected = ['Casa con uso de suelo prueba', 'afsdf', 'Locales en Venta Edificio Roble en San Pedro Garza Garcia',  #place expected result with current filters
                    'Locales en Venta Amadeus Calzada del Valle', 'Locales Comerciales en Venta Vasconcelos San Pedro Garza Garcia', 
                    'Local en Renta Plaza Villas Valle San Pedro Garza Garcia', 
                    'Departamento en Renta Torre Arcangeles en San Pedro Garza Garcia', 
                    'Oficinas en Venta en Valle Oriente', 'Casa en Renta en Lomas del Valle en San Pedro Garza Garcia', 
                    'Departamentos Amueblados en Renta Valle Oriente', 'Oficinas en Renta en Torre Altha San Pedro Garza Garcia', 
                    'Oficina en Venta Centrito 360 San Pedro Garza Garcia', 
                    'Oficinas en Renta en Colonia del Valle San Pedro Garza Garcia', 
                    'Local en Venta Centrito 360 San Pedro Garza Garcia', 'Departamento Amueblado en Renta Valle San Pedro Garza Garcia',
                    'Casa en Renta en Privanzas San Pedro Garza Garcia', 'Local Comercial en Renta en Plaza Cen 333 San Pedro', 
                    'Departamento Amueblado en Renta Torre Koi Valle Pedro Garza Garcia', 
                    'Local en Renta Plaza Maranta San Pedro Garza Garcia', 'Local Renta Valle']
        self.assertListEqual(getProperties(url,headers,args).printProperties(),expected)  #method to compare the lists
