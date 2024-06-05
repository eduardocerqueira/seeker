#date: 2024-06-05T16:53:47Z
#url: https://api.github.com/gists/bec51e05aace2ec7e89f9d700f845770
#owner: https://api.github.com/users/tanooj-s

# output tangential and normal pressure tensor profiles
# (read in LAMMPS compute stress/spherical output and parse appropriately)

import numpy as np
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="pull lammps computed data into a numpy file")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-R', action="store", dest="radius") # sphere size in angstroms
args = parser.parse_args()
R = float(args.radius)
def purge(tokens): "**********"
with open(args.input,'r') as f: lines = f.readlines()
tensors = [] # trajectory of copmuted profiles
for l in lines[3:]:
    tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"> "**********"  "**********"8 "**********": "**********"
        # grab correct tokens below (look at what's being computed in lammps script)
        # https://docs.lammps.org/compute_stress_curvilinear.html#compute-stress-spherical-command
        tensor_tuple = "**********"
                            float(tokens[2]), # n(r)
                            float(tokens[3]), # Pk_r
                            float(tokens[4]), # Pk_phi
                            float(tokens[5]), # Pk_theta 
                            float(tokens[6]), # Pv_r
                            float(tokens[7]), # Pv_phi
                            float(tokens[8])]) # Pv_theta
        tensor.append(tensor_tuple)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"= "**********"= "**********"  "**********"2 "**********": "**********"
        idx = lines.index(l) # line index
        if idx > 20: # reset
            tensor = np.array(tensor)
            tensors.append(tensor)
            tensor = [] 
        else:
            tensor = [] 
tensors = np.array(tensors)
output = args.input[:-4] + 'npy'
with open(output,'wb') as f: np.save(f, tensors)
    

data = tensors
r = 0.1*data[0,:,0] # angstrom -> nm
n = np.mean(data[:,:,1],axis=0)
tensor = np.mean(data[:,:,2:],axis=0)

Pk_r = tensor[:,0]
Pk_phi = tensor[:,1]
Pk_theta = tensor[:,2]
Pv_r = tensor[:,3]
Pv_phi = tensor[:,4]
Pv_theta = tensor[:,5]
# kinetic components will be 0 if coming from a rerun without velocity data, otherwise will likely be orders of magnitude less than virial components 
PN = Pv_r + Pk_r # normal component
PT = 0.5 * (Pv_phi+Pk_phi+Pv_theta+Pv_theta) # tangential component

# scale the same way as done here https://doi.org/10.1063/1.3701372
PN *= (r/R)**2
PT *= (r/R)**2


# estimate surface tension in mN/m
gamma = np.trapz(x=r,y=(PN-PT)) # \int{dr*P} -> [length] * [force/area] -> [force/length] 
gamma *= 1.01325e-2 # [atm/nm] -> [mN/m]
# TODO might need further normalization (actual number of particles per spherical shell?)
# also unit conversions

print(f"Estimated surface tension: {gamma} mN/m")

# plot out
plt.rcParams['figure.figsize'] = (8,4)
plt.rcParams['font.size'] = 16
plt.plot(r,PT,label='tangential',color='b')
plt.plot(r,PN,label='normal',color='r')
plt.axhline(0,color='k',linestyle='dashed')
plt.xlabel('r (nm)')
plt.ylabel('Pressure (atm)')
plt.xlim(0,3.)
plt.grid()
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{args.output}.pressure_spherical.png")

