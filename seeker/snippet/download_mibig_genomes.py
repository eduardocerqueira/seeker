#date: 2023-02-02T16:54:27Z
#url: https://api.github.com/gists/d5f68caab16cb67da93eaedc9bd39655
#owner: https://api.github.com/users/chasemc

#!/usr/bin/env python3

# script to find and download genomes associated with mibig

from pathlib import Path
import os
import hashlib
import requests
import csv
import argparse
from Bio import Entrez
import json

Entrez.email = os.environ["ENTREZ_EMAIL"]
Entrez.api_key = os.environ["ENTREZ_API"]

parser = argparse.ArgumentParser(
    description="Script to find and download genomes associated with mibig"
)

Entrez.email = os.environ["ENTREZ_EMAIL"]
Entrez.api_key = os.environ["ENTREZ_API"]

parser.add_argument(
    "--tsv_outpath",
    metavar="filepath",
    help="Path to write tsv file linking MIBiG id, to nucleotide id, to assembly id",
    required=True,
)
parser.add_argument(
    "--json_dir",
    metavar="filepath",
    help="Downloaded, untarred, MIBiG json directory (https://dl.secondarymetabolites.org/mibig/mibig_json_3.1.tar.gz)",
    required=False,
)
parser.add_argument(
    "--genome_outdir",
    metavar="filepath",
    help="Directory genomes will be downloaded to",
    required=False,
)
parser.add_argument(
    "--contig_id",
    nargs="+",
    metavar="str",
    help="Search just for an individual contig",
    required=False,
)
parser.add_argument(
    "--endswith",
    metavar="str",
    default="_genomic.gbff.gz",
    help="File to download ends with...",
    required=False,
)


def extract_md5sum_from_filetext(url, filename):
    try:
        md5_req = requests.get(url, stream=True)
        md5 = [i.split("  ") for i in md5_req.text.split("\n")]
        md5 = [i for i in md5 if len(i) == 2]
        md5 = [i for i in md5 if i[1] == filename][0]
        return md5
    except:
        return [["this is will fail md5sum", "this is will fail md5sum"]]


def extract_mibig(json_dir):
    # get all json file paths
    pathlist = Path(json_dir).glob("*.json")
    # read json files and create dict  {mibig_id:locus_accession}
    mibig_dict = {}
    for path in pathlist:
        with open(path, "r") as f:
            data = json.load(f)
            mibig_dict[data["cluster"]["mibig_accession"]] = data["cluster"]["loci"][
                "accession"
            ]
    return mibig_dict


def get_assemblies(
    contig,
    bgc_id=None,
    endswith="_genomic.gbff.gz",
    genome_outdir=None,
    tsv_outpath=None,
):
    esearch_file = Entrez.esearch(db="assembly", term=contig)
    esearch_record = Entrez.read(esearch_file)
    result = ""
    assembly_version = 0
    for id in esearch_record["IdList"]:
        es_handle = Entrez.esummary(db="assembly", id=id, report="full")
        res = Entrez.read(es_handle)
        assembly_accession = res["DocumentSummarySet"]["DocumentSummary"][0][
            "AssemblyAccession"
        ]
        res_version = int(assembly_accession.split(".")[1])
        if res_version > assembly_version:
            assembly_version = res_version
            result = res["DocumentSummarySet"]["DocumentSummary"][0]
    if not result:
        return
    url = result["FtpPath_RefSeq"]
    if url == "":
        return
    label = os.path.basename(url)
    # get the assembly link - change this to get other formats
    link = os.path.join(url, label + endswith)
    link = link.replace("ftp://", "https://", 1)
    md5_url = os.path.join(url, "md5checksums.txt")
    md5_url = md5_url.replace("ftp://", "https://", 1)
    if genome_outdir:
        # download link
        outpath = Path(genome_outdir, f"{label}.gbff.gz")
        genome_req = requests.get(link)
        expected_md5 = extract_md5sum_from_filetext(
            url=md5_url, filename=f"./{label + endswith}"
        )
        if hashlib.md5(genome_req.content).hexdigest() == expected_md5[0]:
            with open(outpath, "wb") as f:
                f.write(genome_req.content)
            with open(Path(genome_outdir, "md5sums"), "a") as f:
                f.writelines(f"{expected_md5[0]}  {expected_md5[1]}\n")
    if tsv_outpath:
        with open(tsv_outpath, "a") as handle:
            tsv_writer = csv.writer(
                handle, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            tsv_writer.writerow([bgc_id, contig, assembly_accession])


def main():
    args = parser.parse_args()

    mibig_contig_dict = []
    if args.json_dir:
        if not args.genome_outdir:
            raise TypeError(
                "If '--json_dir' argument is provided you must use the '--genome_outdir' argument"
            )
        mibig_contig_dict = extract_mibig(args.json_dir)
    elif args.contig_id:
        mibig_contig_dict = ({i, x} for i, x in enumerate(args.contig_id))

    if mibig_contig_dict:
        total_count = len(mibig_contig_dict)
        counter = 1
        for k, v in mibig_contig_dict.items():
            get_assemblies(
                contig=v,
                bgc_id=k,
                endswith=args.endswith,
                genome_outdir=args.genome_outdir,
                tsv_outpath=args.tsv_outpath,
            )
            print(f"{str(counter)}/{total_count}", end="\r")
            counter += 1


if __name__ == "__main__":
    main()
