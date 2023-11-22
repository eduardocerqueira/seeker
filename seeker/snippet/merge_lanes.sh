#date: 2023-11-22T16:55:27Z
#url: https://api.github.com/gists/258ed33a0bca67b8b5296b0dde58b7a6
#owner: https://api.github.com/users/slavailn

#! /bin/bash

# Taken from https://github.com/stephenturner/mergelanes/issues/1
# Exercise caution, does not work accurately in every case:
# Not working accurately for sample IDs like "A11_Barcodexxx_S11_L001_R1_001". 
# It cat together all L001 pertaining to sample ID A11 with different barcodes also

ls *R1* | cut -d _ -f 1 | sort | uniq \
    | while read id; do \
        cat $id*R1*.fastq.gz > $id.R1.fastq.gz;
        cat $id*R2*.fastq.gz > $id.R2.fastq.gz;
      done
