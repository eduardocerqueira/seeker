#date: 2023-03-17T16:52:27Z
#url: https://api.github.com/gists/c7d28da213dc4c787aad6636f745c2b7
#owner: https://api.github.com/users/matteoferla

import os
import pickle, gzip
from typing import Dict, List

# -------------------------------
## Import hits

from rdkit import Chem

with Chem.SDMolSupplier('👾👾👾.sdf') as sds:
    hits: List[Chem.Mol] = list(sds)
    
hit_smileses = [Chem.MolToSmiles(hit) for hit in hits]
hit_names = [hit.GetProp('_Name') for hit in hits]
hitdex = dict(zip(hit_names, hits))
target_name = '👾👾👾'
make_fauxalysis(hits, target_name, os.path.join(os.getcwd(), 'fauxalysis'))

# -------------------------------
## Search

from merge import query
from merge.find_merges import getFragmentNetworkSearcher
from merge.find_merges_generic import MergerFinder_generic  # solely for typehinting

searcher: MergerFinder_generic = getFragmentNetworkSearcher()

valid_smileses, valid_names = searcher.filter_for_nodes(hit_smileses, hit_names)
smiles_pairs, name_pairs = searcher.get_combinations(valid_smileses, valid_names)
all_mergers: List[Dict] = []
for smiles_pair, name_pair in zip(smiles_pairs, name_pairs):
    mergers: Dict[str, List[str]] = searcher.get_expansions(smiles_pair, name_pair, target_name, 'output')
    all_mergers.append(dict(mergers=mergers, smiles_pair=smiles_pair, name_pair=name_pair))

with gzip.open('👾👾👾.pkl.gz', 'wb') as fh:
    pickle.dump(all_mergers, fh)
    
print(len(all_mergers),\
sum([len(m['mergers']) for m in all_mergers]), \
sum([len(mm) for m in all_mergers for mm in m['mergers'].values()]))

# -------------------------------
## Parse synthons

import operator
import pandas as pd
from fragmenstein import Victor, Laboratory, Igor

dfs = [ pd.DataFrame([{'smiles': synthon.replace('Xe', 'H'),
                       'original_name': f'{merge_info["name_pair"][1]}-synthon{i}',
                       'xenonic': synthon,
                       'parent': merge_info['name_pair'][1],
                       'hits': [hitdex[merge_info['name_pair'][1]]]} for i, synthon in enumerate(merge_info['mergers'].keys())])
       for merge_info in all_mergers
      ]

synthons = pd.concat(dfs, axis='index')

# fix duplicated
synthons['inchi'] = synthons.smiles.apply(Chem.MolFromSmiles).apply(Chem.RemoveAllHs).apply(Chem.MolToInchiKey)    
synthons = synthons.drop_duplicates(['parent', 'inchi'])
synthons['name'] = synthons.parent +'§'+ (synthons.groupby(['parent']).cumcount()+1).astype(str)
Igor.init_pyrosetta()
placed_synthons = Laboratory(pdbblock=pdb_block, covalent_resi=None).place(synthons, n_cores=2)

def fix_name(row):
    # error... min_mol has it. not unmin.
    mol = Chem.Mol(row.unmin_binary)
    mol.SetProp('_Name', row['name'])
    return mol
    
synthons['∆∆G'] = placed_synthons['∆∆G']
synthons['unmin_mol'] = placed_synthons.apply(fix_name, axis=1)
from rdkit.Chem import PandasTools
PandasTools.WriteSDF(df=synthons,
                     out='👾👾👾-synthons.sdf',
                     molColName='unmin_mol', 
                     idName='name',
                     properties=['parent', '∆∆G'])
## --------------------------------------
# fix names of synthons in combination and make it a long table
data = []
combodex: dict
for combodex in all_mergers:
    # 'mergers', 'smiles_pair', 'name_pair'
    first_name, second_name = combodex['name_pair']
    first: Chem.Mol = hitdex[first_name]
    for synthon_smiles in combodex['mergers']:
        clean_smiles = synthon_smiles.replace('Xe', 'H')
        inchi = Chem.MolToInchiKey( Chem.RemoveAllHs( Chem.MolFromSmiles(clean_smiles) ) )
        matched = placed_synthons.loc[(placed_synthons['parent'] == second_name) & (placed_synthons.inchi == inchi)]
        if len(matched) == 0:
            print(first_name, second_name, synthon_smiles, 'missing!')
            # Z2111637360
            second = hitdex[second_name]
            synthon_name = second_name+'§X'
        elif matched.iloc[0]['∆∆G'] > -1.:
            # skip crap floater fragments
            continue
        else:
            second = matched.iloc[0].unmin_mol
            synthon_name = matched.iloc[0]['name']
        for i, smiles in enumerate(combodex['mergers'][synthon_smiles]):
            name = f'{first_name}-{synthon_name}-{i}'
            data.append(dict(name=name, hits=[first, second], 
                             primary_name=first_name, secondary_parent=second_name, secondary_name=synthon_name,
                             smiles=smiles.replace('Xe', 'H')))
tabular_combinations = pd.DataFrame(data)

# -------------------------
## Place enumerations
    
lab = Laboratory(pdb_block, None)
Victor.monster_throw_on_discard = True
placed = lab.place(tabular_combinations, n_cores=20, expand_isomers=True)
with gzip.open('👾👾👾.placed.pkl.gz', 'wb') as fh:
  placed.to_pickle(fh)
placed.sort_values('∆∆G', ascending=True)
