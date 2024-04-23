#date: 2024-04-23T17:05:33Z
#url: https://api.github.com/gists/82624ffe6c07894c8c444535c9a5db06
#owner: https://api.github.com/users/shekfeh

import sys
import cirpy
import pubchempy as pcp
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote

def fill_structures_2d(sdf_path, output_sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, strictParsing=False)
    output_mols = []
    failed_molecules = []

    counter = 1  # Counter for updated structures

    for mol in suppl:
        if mol:
            props_dict = mol.GetPropsAsDict()
            print(f"Properties for molecule: {props_dict.keys()}")

            if 'INCHIKEY' in props_dict:
                inchi_key = props_dict['INCHIKEY']
                if inchi_key:  # Check if INCHIKEY is not an empty string
                    smiles = inchikey_to_smiles(inchi_key)
                    if not smiles:
                        if 'NAME' in props_dict:
                            name = props_dict['NAME']
                            if name:  # Check if NAME is not an empty string
                                smiles = name_to_smiles(name)
                                if not smiles:
                                    if 'CASNO' in props_dict:
                                        cas = props_dict['CASNO']
                                        if cas:  # Check if CASNO is not an empty string
                                            smiles = cas_to_smiles(cas)
                                            if not smiles:
                                                print("Failed to obtain SMILES for molecule")
                                                failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
                                                continue
                                            else:
                                                failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
                                                continue
            elif 'NAME' in props_dict:
                name = props_dict['NAME']
                if name:  # Check if NAME is not an empty string
                    smiles = name_to_smiles(name)
                    if not smiles:
                        if 'CASNO' in props_dict:
                            cas = props_dict['CASNO']
                            if cas:  # Check if CASNO is not an empty string
                                smiles = cas_to_smiles(cas)
                                if not smiles:
                                    print("Failed to obtain SMILES for molecule")
                                    failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
                                    continue
                        elif not smiles:
                            print("Failed to obtain SMILES for molecule")
                            failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
                            continue
            elif 'CASNO' in props_dict:
                cas = props_dict['CASNO']
                if cas:  # Check if CASNO is not an empty string
                    smiles = cas_to_smiles(cas)
                    if not smiles:
                        print("Failed to obtain SMILES for molecule")
                        failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
                        continue
        else:
            print("Neither INCHIKEY nor NAME nor CASNO found for molecule")
            output_mols.append(mol)
            continue

        if smiles:
            newmol = generate_2d_mol(smiles)
        else:
            print(f"No Smiles was generated")
            failed_molecules.append(props_dict.get('Molecule Name', 'Unnamed Molecule'))
            continue

        if newmol:
            # Preserve properties
            copy_properties(mol, newmol)
            # Update the structure in the original SDF with the retrieved structure
            mol = newmol
            print(f"Updated structure #{counter}")
            counter += 1  # Increment the counter
            output_mols.append(mol)

    # Write the updated molecules to the output SDF file
    w = Chem.SDWriter(output_sdf_path)
    for mol in output_mols:
        if mol is not None:  # Add this check
            w.write(mol)
        else:
            print("Error: Encountered NoneType mol, skipping...")
    w.close()
    print(f"Updated SDF written to {output_sdf_path}")

    # Write failed molecules to a separate file
    write_failed_molecules(failed_molecules, input_sdf_file_path)

# Essential Functions

def write_failed_molecules(failed_molecules, input_sdf_file_path):
    with open(f"failed_{input_sdf_file_path}.txt", 'w') as f:
        f.write("Failed Molecules:\n")
        for molecule in failed_molecules:
            f.write(f"{molecule}\n")
    print(f"Failed molecules written to failed_{input_sdf_file_path}.txt")

def copy_properties(original_mol, updated_mol):
    # Copy properties from original molecule to updated molecule
    for prop in original_mol.GetPropNames():
        updated_mol.SetProp(prop, original_mol.GetProp(prop))


def CIRconvert(ids):
    try:
        url = "http://cactus.nci.nih.gov/chemical/structure/" + quote(ids) + "/smiles"
        ans = urlopen(url).read().decode("utf8")
        return ans
    except:
        return "Did not work"

# identifiers  = ["3-Methylheptane", "Aspirin", "Diethylsulfate", "Diethyl sulfate", "50-78-2", "Adamant"]
# for ids in identifiers :
#     print(ids, CIRconvert(ids))

def get_2d_mol_from_inchikey(inchi_key):
    smiles = inchikey_to_smiles(inchi_key)
    if smiles:
        return generate_2d_mol(smiles)
    else:
        return None

def cas_to_smiles(cas):
    try:
        smiles = CIRconvert(cas)
        return smiles
    except Exception as e:
        print(f"Error converting the CAS num {cas} to SMILES: {e}")
        return None

def name_to_smiles(name):
    try:
        compound = pcp.get_compounds(name, "name")[0]
        smiles = compound.canonical_smiles
        return smiles
    except Exception as e:
        print(f"Error converting the chemical name {name} to SMILES: {e}")
        return None

def inchikey_to_smiles(inchi_key):
    try:
        compound = pcp.get_compounds(inchi_key, "inchikey")[0]
        smiles = compound.canonical_smiles
        return smiles
    except Exception as e:
        print(f"Error converting InChIKey {inchi_key} to SMILES: {e}")
        return None

def generate_2d_mol(smiles):
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol:
            return mol
        else:
            print(f"Failed to generate 2D mol for SMILES: {smiles}")
            return None
    except Exception as e:
        print(f"Error generating 2D mol for SMILES {smiles}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fill_structure.py <filename>")
        sys.exit(1)

    input_sdf_file_path = sys.argv[1]
    output_sdf_file_path = f"./output_{input_sdf_file_path}"
    fill_structures_2d(input_sdf_file_path, output_sdf_file_path)
