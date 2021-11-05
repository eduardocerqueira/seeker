#date: 2021-11-05T16:54:37Z
#url: https://api.github.com/gists/2ec054155dc3a523321253b1d5555da1
#owner: https://api.github.com/users/Sara-Londono

#!/bin/bash 

## Microbial community diversity workflow using 16S V4 region Qiime 2 
## Author Sara Londoño 20211021


## BEFORE YOU START
## Download Qiime2 and set it in a new enviroment (https://docs.qiime2.org/2021.8/install/)
## Make sure you are inside conda Qiime2 enviroment
## you must create your own manifest file
## All .qza and .qzv output files can be visualized in https://view.qiime2.org/

## About manifest file or METADATA file
## In this specific case, the first column tittle must be "sample-id", follow by "absolute-filepath"
# the other columns could be "SampleType" or other metadata you are interest in


# Convert METADATA into a .qzv file 
    qiime metadata tabulate \
     --m-input-file METADATA.tsv\
     --o-visualization Output_Files/METADATA-tabulated.qzv

# Importing data
    qiime tools import \
     --type 'SampleData[SequencesWithQuality]' \
     --input-path METADATA.tsv \
     --output-path Output_Files/single-end-6s-demux.qza \
     --input-format SingleEndFastqManifestPhred33V2

    # Visualize the output - This allows you to determine how many sequences were obtained per sample, 
    # and also to get a summary of the distribution of sequence qualities at each position in your 
    # sequence data.
     qiime demux summarize \
     --i-data Output_Files/single-end-6s-demux.qza \
     --o-visualization Output_Files/single-end-6s-demux.qzv

# Denoising and filtering to get ASVs (Amplicon sequence variants) using DADA2
# to set the parameters follow  this example https://docs.qiime2.org/2021.8/tutorials/moving-pictures/
    qiime dada2 denoise-single \
     --i-demultiplexed-seqs Output_Files/single-end-6s-demux.qza \
     --p-trim-left-f 0 \
     --p-trim-left-r 0 \
     --p-trunc-len-f 150 \
     --p-trunc-len-r 150 \
     --o-table Output_Files/ASV/single-end-6s-table.qza \
     --o-representative-sequences Output_Files/ASV/single-end-6s-rep-seqs.qza \
     --o-denoising-stats Output_Files/ASV/single-end-6s-stats.qza

    # Convert table, rep-seqs and stats in a .qzv files
    qiime feature-table summarize \
     --i-table Output_Files/ASV/single-end-6s-table.qza\
     --o-visualization Output_Files/ASV/single-end-6s-table.qzv \
     --m-sample-metadata-file METADATA.tsv

    qiime feature-table tabulate-seqs \
     --i-data Output_Files/ASV/single-end-6s-rep-seqs.qza \
     --o-visualization Output_Files/ASV/single-end-6s-rep-seqs.qzv
  
    qiime metadata tabulate \
     --m-input-file Output_Files/ASVs/dada2-stats.qza \
     --o-visualization Output_Files/ASVs/dada2-stats.qzv

     # Export the representative sequences into a fasta file, to have them handy.
    qiime tools export \
     --input-path Output_Files/ASV/single-end-6s-rep-seqs.qza \
     --output-path Output_Files/ASV/rep-seqs
