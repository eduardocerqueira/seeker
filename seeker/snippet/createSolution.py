#date: 2021-11-10T16:51:40Z
#url: https://api.github.com/gists/cb57d22178957292a34ec9c7373140b5
#owner: https://api.github.com/users/tanooj-s

# create a data file to be read in by LAMMPS
# random config of HCl solution
# H2O molecules, Na+ ions, Cl- ions
# do an NaCl solution instead

# need to make sure molecules are randomly oriented as well

# take in rho (number density) and threshold distance from user

# need TIP3P for CHARMM instead
# SPC constraints
# HOH bond angle = 109.47 degrees
# OH bond length = 1 A

# METAL UNITS



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import time
import os



plt.rcParams['figure.figsize'] = (20,15)
plt.rcParams['font.size'] = 22

parser = argparse.ArgumentParser(description="Random config to create a solution")
parser.add_argument('-rho', action='store', dest='rho') # between 0.0 and 1.0, number density
parser.add_argument('-t', action='store', dest='threshold') # no contact radius
args = parser.parse_args()

class Atom:
	def __init__(self,atom_id,mol_id,atom_type,charge,x,y,z):
		self.atom_id = atom_id
		self.mol_id = mol_id
		self.type = atom_type
		self.charge = charge
		self.position = np.array([x,y,z])

threshold = float(args.threshold)
rho = float(args.rho)

n_mols = 1000

n_atoms = 3*n_mols # this will change

v = n_atoms/rho
dim = np.power(v,1/3)
L = np.array([dim,dim,dim]) # cubic box
print("Box dimensions: ")
print(L)

atoms = [] # list of atom objects to be written to file

# place the cavity atom at the center
start = time.time()

r = 1 # OH bond length in angstroms
theta = 109.47 # HOH bond angle in degrees
phi = 90 - theta/2 # geometry to attach H atoms 

# masses in AMU - get the exact values
m_H = 1
m_O = 16 
m_Cl = 35
m_Na = 22

# need to make sure system is charge neutral 

# function to randomly orient water molecule, take in position of oxygen atom as input, set as origin
# generalize this function to take in any initial structure and return 

def rotate_molecule(initial_positions, origin):
	# randomly chosen euler angles to rotate molecule by
	alpha = np.random.uniform(0,2*np.pi)
	beta = np.random.uniform(0,2*np.pi)
	gamma = np.random.uniform(0,2*np.pi)

	rotated_positions = []

	matrix = np.array([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)],
				   [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)],
				   [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])

	for vector in initial_positions:
		rotated_positions.append(origin + np.matmul(matrix,vector))

	return rotated_positions

# initial positions for water molecule 
# O: 0.00 0.00 0.00
# H: 0.816490 0.577736 0.00
# H: -0.816490 0.577736 0.00

# initial positions for hydronium 
# O: 0.000000 0.000000 0.000000
# H: 0.000000 0.939693 -0.342020
# H: 0.813798 -0.469846 -0.342020
# H: -0.813798 -0.469846 -0.342020

h2o_geom = np.array([[0.00, 0.00, 0.00],
					[0.81649,0.577736,0.00],
					[-0.81649,0.577736,0.00]])

h3o_geom = np.array([[0.00, 0.00, 0.00],
					[0.00, 0.939693, -0.342020],
					[0.813798,-0.469846,-0.342020],
					[-0.813798,-0.469846,-0.342020]])

n_h2o = 200 # number of h2o molecules
n_na = 4 # number of Na+ ions
n_cl = 4 # number of Cl- ions
# modify latter two according to correct mole fraction



# clean this up so that you don't have to repeat so much code


atom_counter = 1 # index of each atom being placed

print("Placing H2O molecules...")
current_mol = 1 # index of molecule being placed
max_attempts = 10 # log max attempts it takes to place an atom, will be near the end
while current_mol <= n_h2o:
	is_placed = False
	n_attempts = 0
	while is_placed == False:
		attempt = np.array([np.random.uniform(0, L[0]), np.random.uniform(0, L[1]), np.random.uniform(0, L[2])])
		n_attempts += 1
		if n_attempts > max_attempts:
			max_attempts = n_attempts 
		n_checked = 0
		# iterate through every atom already placed
		for a in atoms:
			displacement = attempt - a.position
			# PBCs
			while(displacement[0] > L[0]/2):
				displacement[0] -= L[0]
			while(displacement[0] < -L[0]/2):
				displacement[0] += L[0]
			while(displacement[1] > L[1]/2):
				displacement[1] -= L[1]
			while(displacement[1] < -L[1]/2):
				displacement[1] += L[1]
			while(displacement[2] > L[2]/2):
				displacement[2] -= L[2]
			while(displacement[2] < -L[2]/2):
				displacement[2] += L[2]
			distance = np.sqrt(np.dot(displacement,displacement))		
			if (distance < threshold): # need a new attempted position
				break
			n_checked += 1
			#if (mirror_distance < threshold):
			#	break
		if n_checked >= len(atoms): # i.e. every single atom that's in the box has been checked against and not been under the threshold
			o_position = attempt # position of O atom
			atoms.append(Atom(atom_id = atom_counter,
							mol_id = current_mol,
			  			    atom_type = 2,
			  			    charge = -0.82, # SPC param
				  			x = o_position[0],
				  			y = o_position[1],
				  			z = o_position[2]))
			is_placed = True
			atom_counter += 1
			
			#print("Placed atom %s of type %s with charge %s after %s attempts"%(str(new_atom.id), str(new_atom.type), str(new_atom.charge), str(n_attempts)))

			# directly attach 2 H atoms now, rotated wrt to O position
			h1_position, h2_position = rotate_molecule(h2o_geom,o_position)[1:]
			atoms.append(Atom(atom_id = atom_counter,
							mol_id = current_mol,
							atom_type = 1,
							charge = 0.41, # SPC param
							x = h1_position[0],
							y = h1_position[1],
							z = h1_position[2]))
			atom_counter += 1
			atoms.append(Atom(atom_id = atom_counter,
							mol_id = current_mol,
							atom_type = 1,
							charge = 0.41, # SPC param
							x = h2_position[0],
							y = h2_position[1],
							z = h2_position[2]))
			atom_counter += 1
			print("Placed H2O molecule " + str(current_mol))
			print("Attempts: " + str(n_attempts))
	current_mol += 1
# H2O effective charge = 2*0.41 - 0.82 = 0.00


# hydronium structure from http://web.missouri.edu/~glaserr/433f02/hydroniumG1.log

#    1          8           0.000000    0.000000    0.093278
#    2          1           0.000000    0.939693   -0.248742
#    3          1           0.813798   -0.469846   -0.248742
#    4          1          -0.813798   -0.469846   -0.248742
# set the O at (0,0,0) instead
# 1 8 0.000000 0.000000 0.000000
# 2 1 0.000000 0.939693 -0.342020
# 3 1 0.813798 -0.469846 -0.342020
# 4 1 -0.813798 -0.469846 -0.342020

while current_mol <= (n_h2o + n_na):
	is_placed = False
	n_attempts = 0
	while is_placed == False:
		attempt = np.array([np.random.uniform(0, L[0]), np.random.uniform(0, L[1]), np.random.uniform(0, L[2])])
		n_attempts += 1
		if n_attempts > max_attempts:
			max_attempts = n_attempts 
		n_checked = 0
		# iterate through every atom already placed
		for a in atoms:
			displacement = attempt - a.position
			# PBCs
			while(displacement[0] > L[0]/2):
				displacement[0] -= L[0]
			while(displacement[0] < -L[0]/2):
				displacement[0] += L[0]
			while(displacement[1] > L[1]/2):
				displacement[1] -= L[1]
			while(displacement[1] < -L[1]/2):
				displacement[1] += L[1]
			while(displacement[2] > L[2]/2):
				displacement[2] -= L[2]
			while(displacement[2] < -L[2]/2):
				displacement[2] += L[2]
			distance = np.sqrt(np.dot(displacement,displacement))		
			if (distance < threshold): # need a new attempted position
				break
			n_checked += 1
			#if (mirror_distance < threshold):
			#	break
		if n_checked >= len(atoms): # i.e. every single atom that's in the box has been checked against and not been under the threshold
			o_position = attempt # position of O atom
			atoms.append(Atom(atom_id = atom_counter,
							mol_id = current_mol,
			  			    atom_type = 3,
			  			    charge = 1, 
				  			x = o_position[0],
				  			y = o_position[1],
				  			z = o_position[2]))
			is_placed = True
			atom_counter += 1
			#print("Placed atom %s of type %s with charge %s after %s attempts"%(str(new_atom.id), str(new_atom.type), str(new_atom.charge), str(n_attempts)))
			# directly attach 2 H atoms now, rotated wrt to O position
			#h1_position, h2_position, h3_position = rotate_molecule(h3o_geom,o_position)[1:]
			#atoms.append(Atom(atom_id = atom_counter,
			#				mol_id = current_mol,
			#				atom_type = 3,
			#				charge = 0.518, # SPC param
			#				x = h1_position[0],
			#				y = h1_position[1],
			#				z = h1_position[2]))
			#atom_counter += 1
			#atoms.append(Atom(atom_id = atom_counter,
			#				mol_id = current_mol,
			#				atom_type = 3,
			#				charge = 0.518, # SPC param
			#				x = h2_position[0],
			#				y = h2_position[1],
			#				z = h2_position[2]))
			#atom_counter += 1
			#atoms.append(Atom(atom_id = atom_counter,
			#				mol_id = current_mol,
			#				atom_type = 3,
			#				charge = 0.518, # SPC param
			#				x = h3_position[0],
			#				y = h3_position[1],
			#				z = h3_position[2]))
			#atom_counter += 1
			print("Placed Na+ ion " + str(current_mol))
			print("Attempts: " + str(n_attempts))
	current_mol += 1

# H3O+ efffective charge = 2*0.518 - 0.554 = +0.482

# now Cl-
while current_mol <= (n_h2o + n_na + n_cl):
	is_placed = False
	n_attempts = 0
	while is_placed == False:
		attempt = np.array([np.random.uniform(0, L[0]), np.random.uniform(0, L[1]), np.random.uniform(0, L[2])])
		n_attempts += 1
		if n_attempts > max_attempts:
			max_attempts = n_attempts 
		n_checked = 0
		# iterate through every atom already placed
		for a in atoms:
			displacement = attempt - a.position
			# PBCs
			while(displacement[0] > L[0]/2):
				displacement[0] -= L[0]
			while(displacement[0] < -L[0]/2):
				displacement[0] += L[0]
			while(displacement[1] > L[1]/2):
				displacement[1] -= L[1]
			while(displacement[1] < -L[1]/2):
				displacement[1] += L[1]
			while(displacement[2] > L[2]/2):
				displacement[2] -= L[2]
			while(displacement[2] < -L[2]/2):
				displacement[2] += L[2]
			distance = np.sqrt(np.dot(displacement,displacement))		
			if (distance < threshold): # need a new attempted position
				break
			n_checked += 1
			#if (mirror_distance < threshold):
			#	break
		if n_checked >= len(atoms): # i.e. every single atom that's in the box has been checked against and not been under the threshold
			position = attempt # position of O atom
			atoms.append(Atom(atom_id = atom_counter,
							mol_id = current_mol,
			  			    atom_type = 4,
			  			    charge = -1.0, # SPC param
				  			x = position[0],
				  			y = position[1],
				  			z = position[2]))
			is_placed = True
			atom_counter += 1
			print("Placed Cl- ion " + str(current_mol))
			print("Attempts: " + str(n_attempts))
			#print("Placed atom %s of type %s with charge %s after %s attempts"%(str(new_atom.id), str(new_atom.type), str(new_atom.charge), str(n_attempts)))
	current_mol += 1

# now need to add bonds and angles
bonds = []
angles = []

# create a hash map that gives you corresponding atom_ids for every molecule_id
mol_map = dict()
for atom in atoms:
	if atom.mol_id not in mol_map.keys():
		mol_map[atom.mol_id] = [atom.atom_id]
	else:
		mol_map[atom.mol_id].append(atom.atom_id)

print(mol_map)
#exit()


# only one bond type and angle type because same bonded interaction constraints for H2O and H3O+
# angles are easier to deal with so do that first
angle_counter = 1
bond_counter = 1

# generalize this bit - make it a function that defines all bonds and all angles for mixed n-atomic molecules
for mol_id, atom_ids in mol_map.items():
	if len(atom_ids) == 3: # triatomic i.e. H2O
		angles.append([angle_counter, 1, atom_ids[0], atom_ids[1], atom_ids[2]])
		angle_counter += 1
		bonds.append([bond_counter, 1, atom_ids[0], atom_ids[1]])
		bond_counter += 1
		bonds.append([bond_counter, 1, atom_ids[0], atom_ids[2]])
		bond_counter += 1

	# angle type for hydronium should be different because the equilbrium angle is not the same as H2O
	if len(atom_ids) == 4: # hydronium
		angles.append([angle_counter,1,atom_ids[0],atom_ids[1],atom_ids[2]])
		angle_counter += 1
		angles.append([angle_counter,1,atom_ids[0],atom_ids[2],atom_ids[3]])
		angle_counter += 1
		angles.append([angle_counter,1,atom_ids[0],atom_ids[3],atom_ids[1]])
		angle_counter += 1
		bonds.append([bond_counter, 1, atom_ids[0], atom_ids[1]])
		bond_counter += 1
		bonds.append([bond_counter, 1, atom_ids[0], atom_ids[2]])
		bond_counter += 1
		bonds.append([bond_counter, 1, atom_ids[0], atom_ids[3]])
		bond_counter += 1
		# three harmonic angles in each hydronium ion


print(bonds)
print(angles)





end = time.time()
print("%s atoms placed in box with MC-style volume filling"%(str(len(atoms))))
print("Maximum attempts to place a single atom: %s"%(str(max_attempts)))
print("%s s\n"%(str(end-start)))
print("\n================ CHECKING DISTANCES =================\n")
# confirm no distances are under the threshold
# need to do these checks with mirror images as well
n_under = 0
print("Checking no distances are below threshold...")
for atom_i in atoms:
	for atom_j in atoms:
		if atom_i.mol_id != atom_j.mol_id:
			difference = atom_i.position - atom_j.position
			distance = np.sqrt(np.dot(difference,difference))
			if distance < threshold:
				print("\nAtoms listed below are too close")
				print(atom_i.atom_id)
				print(atom_j.atom_id)
				print(atom_i.position)
				print(atom_j.position)
				print(distance)
				print("\n")
				n_under += 1
print("Total pairs of atoms: " + str(n_atoms*(n_atoms-1)/2))
print("Pairs too close: " + str(n_under) + '\n')


# confirm charge neutrality
q_tot = 0
for atom in atoms:
	q_tot += atom.charge

print("Total charge in system: " + str(q_tot))


# write atom data to file 
with open('solution.sparse',"w") as f:
	f.write("Random configuration data file with minimum distance = %s angstroms\n\n"%(str(threshold)))
	f.write("4 atom types\n")
	f.write("1 bond types\n")
	f.write("1 angle types\n")
	f.write("%s atoms\n"%(str(len(atoms))))
	f.write("%s bonds\n"%(str(len(bonds))))
	f.write("%s angles\n"%(str(len(angles))))
	f.write("0 %s xlo xhi\n"%(str(L[0])))
	f.write("0 %s ylo yhi\n"%(str(L[1])))
	f.write("0 %s zlo zhi\n\n"%(str(L[2])))
	f.write("Masses\n\n")
	f.write("1 %s\n"%(str(m_H)))
	f.write("2 %s\n"%(str(m_O)))
	f.write("3 %s\n"%(str(m_Na)))
	f.write("4 %s\n"%(str(m_Cl)))
	f.write("\n")
	f.write("Atoms # full\n")
	f.write("\n")
	for a in atoms:
		# full atom type in LAMMPS
		# atom-id, mol-id, atom-type, q, x, y, z
		f.write("%s %s %s %s %s %s %s\n"%(str(a.atom_id),
										str(a.mol_id), # mol id, need to hack for solute and cavity bond
										str(a.type),
										str(a.charge), # for charge
										str(a.position[0]),
										str(a.position[1]),
										str(a.position[2]))) # index, type, 0(?), x, y, z
	f.write("\n")
	f.write("Bonds\n")
	f.write("\n")
	for b in bonds:
		f.write("%s %s %s %s\n"%(str(b[0]),str(b[1]),str(b[2]),str(b[3])))
	f.write("\n")
	f.write("Angles\n")
	f.write("\n")
	for a in angles:
		f.write("%s %s %s %s %s\n"%(str(a[0]),str(a[1]),str(a[2]),str(a[3]),str(a[4])))

	
print("Random configuration written to solution.sparse\n\n\n\n\n")

# NEED TO ADD BOND INFORMATION AS WELL


# now run a LAMMPS script to compress the solution to atmospheric pressure
