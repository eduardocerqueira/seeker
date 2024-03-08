#date: 2024-03-08T17:07:51Z
#url: https://api.github.com/gists/17f9054ccf24123f6878cd9dc7722179
#owner: https://api.github.com/users/dnanto

#!/usr/bin/env python3

__author__ = "dnanto"

import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
from pathlib import Path

from Bio.PDB import PDBParser, Superimposer


def parse_args(argv):
    parser = ArgumentParser(
        "rmsd",
        description="calculate RMSD between two PDB structures",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path1", type=Path, help="the path to PDB file 1")
    parser.add_argument("path2", type=Path, help="the path to PDB file 2")
    parser.add_argument("-permissive", action="store_true", help="the flag to enable permissive parsing")
    parser.add_argument("-out", type=FileType("w"), default="-", help="the output file stream")
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    parser = PDBParser(PERMISSIVE=args.permissive)
    id1, id2 = args.path1.stem, args.path2.stem
    s1 = parser.get_structure(id1, args.path1)
    s2 = parser.get_structure(id2, args.path2)

    # assume both structures have the same number of chains
    chains = {}
    poser = Superimposer()
    for idx, ele in enumerate(zip(s1.get_chains(), s2.get_chains()), start=1):
        ch1, ch2 = ele
        # get chain id or use iteration number if not defined
        chain_id = (ch1.id or ch2.id) or idx
        ca1, ca2 = (
            [res["CA"] for res in ch.get_residues() if "CA" in res]
            for ch in (ch1, ch2)
        )
        # scan the shorter structure over the longer one
        if len(ca2) < len(ca1):
            id1, id2 = id2, id1
            ca1, ca2 = ca2, ca1
        diff = len(ca2) - len(ca1) + 1
        results = []
        for i in range(diff):
            poser.set_atoms(ca1, ca2[i:(len(ca1) + i)])
            results.append((i, poser.rms))
        i, rmsd = min(results, key=lambda row: row[-1])
        chains[chain_id] = dict(id1=id1, id2=id2, pos1=i + 1, pos2=len(ca1) + i + 1, rmsd=rmsd)

    json.dump(chains, args.out, indent=True)
    args.out.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
