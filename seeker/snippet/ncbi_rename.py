#date: 2025-04-25T17:11:59Z
#url: https://api.github.com/gists/e774e505b73038b743b49520114c5aa1
#owner: https://api.github.com/users/brantfaircloth

#!/usr/bin/env python
# encoding: utf-8
"""
Created by Brant Faircloth on April 24, 2025 at 13:16:35 CDT
Copyright (c) 2025 Brant C. Faircloth. All rights reserved.

Description: Rename a NCBI genome using its assembly_report.

"""

import pdb

from pathlib import Path

import typer
from typing_extensions import Annotated
from rich.progress import track

from Bio import SeqIO



def main(genome: Annotated[Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
report: Annotated[Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
output: Annotated[Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=False,
            resolve_path=True,
        ),
    ],
alias: Annotated[Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=False,
            resolve_path=True,
        ),
    ],
):
    names = {}
    with open(report) as infile:
        for line in infile:
            if not line.startswith("#"):
                lss = line.strip().split("\t")
                names[lss[6]] = {
                    "chr":lss[2],
                    "genbank":lss[4],
                    "length":int(lss[-2]),
                    "role":lss[1],
                    "type":lss[3]
                }
    with open(genome) as infile:
        with open(output, "w") as outfile:
            # output a chromalias file:
            with open(alias, "w") as chromalias:
                chromalias.write("# ucsc\tassembly\tgenbank\trefseq\n")
                total = 0
                fasta = SeqIO.parse(genome, "fasta")
                for seq in track(fasta, description="Processing..."):
                    # make sure scaff we're on is expected length
                    assert len(seq) == names[seq.id]["length"]
                    # each of these classes get slightly different names
                    if names[seq.id]['type'] == "Chromosome" and names[seq.id]['role'] == "assembled-molecule":
                        new_name = f"chr{names[seq.id]['chr']}"
                    elif names[seq.id]['type'] == "Chromosome" and names[seq.id]['role'] == "unlocalized-scaffold":
                        new_name = f"chr{names[seq.id]['chr']}_{seq.id.replace('.', 'v')}"
                    elif names[seq.id]['type'] == "na" and names[seq.id]['role'] == "unplaced-scaffold":
                        new_name = f"chrUn_{seq.id.replace('.', 'v')}"
                    elif names[seq.id]['type'] == "Mitochondrion" and names[seq.id]['role'] == "assembled-molecule":
                        new_name = "chrM"
                    else:
                        raise(IOError, "The `type` or `role` of the scaffold|contig in the report is unexpected.")
                    # create the chrom_alias entry for this scaffold
                    chromalias.write(f"{new_name}\t{names[seq.id]['chr']}\t{names[seq.id]['genbank']}\t{seq.id}\n")
                    # finally, rewrite the fasta w/ new name
                    seq.id = new_name
                    seq.name = ""
                    seq.description = ""
                    outfile.write(seq.format("fasta"))
                    total += 1
                print(f"Processed {total} entries.")



if __name__ == "__main__":
    typer.run(main)