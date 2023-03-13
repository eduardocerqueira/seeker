#date: 2023-03-13T17:06:21Z
#url: https://api.github.com/gists/23646c8cce64d4cdfe61467da64853c0
#owner: https://api.github.com/users/Gus-The-Forklift-Driver

node = hou.pwd()
geo = node.geometry()

# need packed geo to have a prim attribute 'name' contaning the name of the prim
# need packed geo to have a prim attribute 'transforms' containing the matrix transform of the packed prim

with open(node.evalParm('path'),'w') as file:
    for prim in geo.prims():
        transform = prim.attribValue('transforms')
        for x in transform:
            file.write(str(x) + ',')
        file.write(prim.attribValue('name')+'\n')
