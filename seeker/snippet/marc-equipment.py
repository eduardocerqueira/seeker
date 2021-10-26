#date: 2021-10-26T17:15:16Z
#url: https://api.github.com/gists/02b72e9a683d5146b599939479a06ab2
#owner: https://api.github.com/users/jamesrf

import pymarc
import csv

marcfile = "evergreen-full-latest.mrc"

circ_modifiers = ['laptop','tablet','library-equipment']


with open(marcfile, 'rb') as data, open('output.csv','w', newline='') as csvfile:
    reader = pymarc.MARCReader(data)
    writer = csv.writer(csvfile)
    for record in reader:
        holdings = record.get_fields('852')

        for item in holdings:
            circ_modifier = item['g']

            if circ_modifier not in circ_modifiers:
                continue

            circ_lib = item['b'][7:11]
            title = record.title()
            item_number = item['t']
            call_number = item['j']
            name = call_number + " #" + item_number
            
            barcode = item['p']
            price = item['y']

            model = [ m['a'] for m in record.get_fields('590') ]
            model = ";".join(model)

            description = ";".join([ d['a'] for d in record.get_fields('520','538')])

            manual = record.get_fields('856')
            if manual:
                manual = manual[0]['u']
            else:
                manual = ""

            writer.writerow([circ_lib, title, name, description, barcode, model, price, manual])
            print('.')
            



print("done")