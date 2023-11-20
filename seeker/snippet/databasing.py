#date: 2023-11-20T16:44:34Z
#url: https://api.github.com/gists/4941233d7331604fe9dce3e64674fba0
#owner: https://api.github.com/users/andrewtarzia


"""
MIT License
Copyright (c) 2023 Andrew Tarzia
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import stk
import atomlite
import stko


def main():
    db = atomlite.Database("test.db")
    print(db.num_entries())

    # Define an stk molecule and build a cage.
    bb1 = stk.BuildingBlock(
        smiles="NCCN",
        functional_groups=[stk.PrimaryAminoFactory()],
    )
    bb2 = stk.BuildingBlock(
        smiles="O=CC(C=O)C=O",
        functional_groups=[stk.AldehydeFactory()],
    )

    cage = stk.ConstructedMolecule(
        topology_graph=stk.cage.FourPlusSix(
            building_blocks=(bb1, bb2),
            optimizer=stk.MCHammer(),
        ),
    )

    # Add them all.
    entries = [
        atomlite.Entry.from_rdkit(
            key="cage",
            molecule=cage.to_rdkit_mol(),
            properties={"is_building_block": False},
        ),
        atomlite.Entry.from_rdkit(
            key="bb1",
            molecule=bb1.to_rdkit_mol(),
            properties={"is_building_block": True},
        ),
        atomlite.Entry.from_rdkit(
            key="bb2",
            molecule=bb2.to_rdkit_mol(),
            properties={"is_building_block": True},
        ),
    ]
    # Use update_entries to avoid an error if the molecule/entry already
    # exists.
    db.update_entries(entries)

    print(db.num_entries())
    input()

    # Iterate through all.
    for entry in db.get_entries():
        # If not building block, analyse.
        if entry.properties["is_building_block"]:
            print(entry)
            continue

        # Add a property dictionary.
        analyser = stko.molecule_analysis.GeometryAnalyser()
        molecule = stk.BuildingBlock.init_from_rdkit_mol(
            atomlite.json_to_rdkit(entry.molecule)
        )
        db.update_properties(
            atomlite.PropertyEntry(
                key=entry.key,
                properties={"pore_size": analyser.get_min_centroid_distance(molecule)},
            )
        )
    input()

    # Iterate through all some time later.
    for entry in db.get_entries():
        print(entry.key, entry.properties)


if __name__ == "__main__":
    main()