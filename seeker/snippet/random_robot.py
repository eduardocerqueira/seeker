#date: 2022-02-24T17:02:33Z
#url: https://api.github.com/gists/3c62ef8aef3c024b13c3ce4bb9505340
#owner: https://api.github.com/users/skriegman

from lxml import etree
import numpy as np
from one_shape import make_one_shape_only  # https://gist.github.com/skriegman/32b9fd470072c0bd8ec69a2fceb39bea

SEED = 0

np.random.seed(SEED)

BUILD_DIR = "/voxcraft-sim/build"

RECORD_HISTORY = True  # saves the behavior movie

bx, by, bz = (8, 8, 8)  # body size

body = np.random.rand(bx, by, bz)  # random floats between 0 and 1

blob = make_one_shape_only(body>0.25)  # one shape where random floats > 0.25

body = np.array(blob, dtype=np.int8)

# don't eval passive bodies
if np.sum(body == 1) == 0:
    print("null")
    exit()

# shift down until in contact with surface plane
while True:
    if np.sum(body[:, :, 0]) == 0:
        body[:, :, :-1] = body[:, :, 1:]
        body[:, :, -1] = np.zeros_like(body[:, :, -1])
    else:
        break

body = np.swapaxes(body, 0,2)
# body = body.reshape([wz,-1])  # you can do it this way too, I think
body = body.reshape(bz, bx*by)

control = np.random.rand(bz, bx*by)  # I am typing this in browser without testing
control = 2*control - 1  # random uniform dist between -1 and 1

# get new voxcraft build
sub.call("cp {}/voxcraft-sim .".format(BUILD_DIR), shell=True)
sub.call("cp {}/vx3_node_worker .".format(BUILD_DIR), shell=True)

# create data folder if it doesn't already exist
sub.call("mkdir data{}".format(SEED), shell=True)
sub.call("cp base.vxa data{}/base.vxa".format(SEED), shell=True)  # you will need a base.vxa

# clear old .vxd robot files from the data directory
sub.call("rm data{}/*.vxd".format(SEED), shell=True)

# delete old hist file
sub.call("rm a{}.hist".format(SEED), shell=True)

# delete old workspace
sub.call("rm -r workspace", shell=True)

# remove old sim output.xml to save new stats
sub.call("rm output{}.xml".format(SEED), shell=True)

# start vxd file
root = etree.Element("VXD")

if RECORD_HISTORY:
    # sub.call("rm a{0}_gen{1}.hist".format(seed, pop.gen), shell=True)
    history = etree.SubElement(root, "RecordHistory")
    history.set('replace', 'VXA.Simulator.RecordHistory')
    etree.SubElement(history, "RecordStepSize").text = '100'
    etree.SubElement(history, "RecordVoxel").text = '1'
    etree.SubElement(history, "RecordLink").text = '0'
    etree.SubElement(history, "RecordFixedVoxels").text = '1'
    etree.SubElement(history, "RecordCoMTraceOfEachVoxelGroupfOfThisMaterial").text = '0'  # draw CoM trace
    

structure = etree.SubElement(root, "Structure")
structure.set('replace', 'VXA.VXC.Structure')
structure.set('Compression', 'ASCII_READABLE')
etree.SubElement(structure, "X_Voxels").text = str(bx)
etree.SubElement(structure, "Y_Voxels").text = str(by)
etree.SubElement(structure, "Z_Voxels").text = str(bz)

data = etree.SubElement(structure, "Data")
for i in range(body.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) for c in body[i]])  # the body doesn't have commas between the voxels
    layer.text = etree.CDATA(str_layer)

data = etree.SubElement(structure, "PhaseOffset")
for i in range(control.shape[0]):
    layer = etree.SubElement(data, "Layer")
    str_layer = "".join([str(c) + ", " for c in control[i]])  # other variables can be floats so they need commas
    layer.text = etree.CDATA(str_layer)

# save the vxd to data folder
with open('data'+str(SEED)+'/bot_0.vxd', 'wb') as vxd:
    vxd.write(etree.tostring(root))

# ok let's finally evaluate all the robots in the data directory

if RECORD_HISTORY:
    sub.call("./voxcraft-sim -i data{0} -o output{0}.xml -f > a{0}.hist".format(SEED), shell=True)
else:
    sub.call("./voxcraft-sim -i data{0} -o output{0}.xml".format(SEED), shell=True)
