#date: 2022-07-11T17:10:40Z
#url: https://api.github.com/gists/76fa206d90a5f66a81f9fc165bf1dd18
#owner: https://api.github.com/users/mjziebarth

# Converts the WSM 2016 CSV table to GeoJSON.
#
# Author: Malte J. Ziebarth (mjz.science@fmvkb.de)
#
# Copyright (C) 2022 Malte J. Ziebarth
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

# Argument parsing:
import argparse
parser = argparse.ArgumentParser(description='Convert WSM 2016 CSV table to GeoJSON')
parser.add_argument('-i',default='wsm2016.csv')
parser.add_argument('-o',default='wsm2016.geojson')
args = parser.parse_args()

# Decode the CSV:
import codecs
with codecs.open(args.i, 'r', encoding='iso-8859-1') as f:
    lines = f.readlines()

table = []
line_lengths = set()
for line in lines:
    table.append(list(str(t).strip() for t in line.split(',')))
    line_lengths.add(len(table[-1]))

# Header with all labels:
header = [str(t) for t in table[0] if len(t) > 0]

# Conversion function for fields of the table:
field_types \
  = {"AZI"      : float,
     "DEPTH"    : float,
     "DATE"     : int,
     "NUMBER"     : int,
     "S1AZ"       : int,
     "S1PL"       : int,
     "S2AZ"       : int,
     "S2PL"       : int,
     "S3AZ"       : int,
     "S3PL"       : int,
     "MAG_INT_S1" : float,
     "SLOPES1"    : float,
     "MAG_INT_S2" : float,
     "SLOPES2"    : float,
     "MAG_INT_S3" : float,
     "SLOPES3"    : float,
     "PORE_MAGIN" : float,
     "PORE_SLOPE" : float
}


# Write GeoJSON:
geojson = '{\n' \
          '  "type": "FeatureCollection",\n' \
          '  "features": [\n'

nrow = len(header)
for j,row in enumerate(table[1:]):
    geojson += '    {\n' \
               '      "type": "Feature",\n' \
               '      "geometry": {\n' \
               '         "type": "Point",\n' \
              f'         "coordinates": [{row[3]},{row[2]}]\n' \
               '      },\n' \
               '      "properties": {\n'
    for i,(h,r) in enumerate(zip(header,row)):
        if h in ('LON','LAT'):
            continue
        if h in field_types:
            try:
                field = field_types[h](r)
            except:
                field = '"' + str(r).replace('"','\\"') + '"'
        else:
            field = '"' + str(r).replace('"','\\"') + '"'
        if i < nrow - 1:
            geojson += f'        "{h}" : {field},\n'
        else:
            geojson += f'        "{h}" : {field}\n'
    geojson += '      }\n' \
               '    }'
    if j < len(table) - 2:
        geojson += ',\n'
    else:
        geojson += '\n'

geojson += '  ]\n}'


with open(args.o, 'w') as f:
    f.write(geojson)

print("success!")