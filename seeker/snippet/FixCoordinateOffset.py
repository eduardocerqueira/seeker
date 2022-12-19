#date: 2022-12-19T16:44:38Z
#url: https://api.github.com/gists/37c32c04c1ee100c9b842af1c70e6295
#owner: https://api.github.com/users/eibre

import ifcopenshell
# Open the IFC file
ifc_file = ifcopenshell.open(r"X:\nor\oppdrag\Molde2\522\04\52204946\BIM\Innsynsmodell\VOS-LARK.ifc")

local_placements = ifc_file.by_type('IfcLocalPlacement')
len(local_placements)

coords = []
for placement in local_placements:
    coords.append(placement.RelativePlacement.Location.Coordinates)
min_coord = min(coords, key=lambda coord: (coord[0], coord[1], coord[2]))
print(min_coord)

points = ifc_file.by_type('IfcPoint')
for point in points:
    if point.Coordinates[0] > abs(min_coord[0]):
        point.Coordinates = (point.Coordinates[0] + min_coord[0], point.Coordinates[1] + min_coord[1], point.Coordinates[2] + min_coord[2])

ifc_file.write(r"C:\Users\eibre\OneDrive - Norconsult Group\Desktop\IFC\VOS-LARK_local.ifc")