#date: 2025-11-26T17:03:13Z
#url: https://api.github.com/gists/f029a9251e48a14bcb4a54fd256e5846
#owner: https://api.github.com/users/chenyongrowan

#!/bin/bash

# Base directories
input_base="/mnt/raid5_array/chenyong_files/XIN_AccessHiC/access-hic_merged"
output_base="/mnt/raid5_array/chenyong_files/XIN_AccessHiC/access-hic_out"

# Reference genome info
genome_idx="/mnt/raid5_array/chenyong_files/XIN_AccessHiC/refrence_genomes/hg38_index"
chrom_sizes="/mnt/raid5_array/chenyong_files/XIN_AccessHiC/refrence_genomes/hg38_index/new_chrom.sizes"

# Path to rowan-hic
rowan_hic="/home/mcsbl/HiCPIP/rowan-hic/bin/rowan-hic"

# Loop through each sample folder
for sample_dir in "$input_base"/*; do
    if [ -d "$sample_dir" ]; then
        sample_name=$(basename "$sample_dir")

        fastq1="$sample_dir/${sample_name}_merged_R01.fastq.gz"
        fastq2="$sample_dir/${sample_name}_merged_R02.fastq.gz"
        output_dir="$output_base/$sample_name"

        # Create output folder
        mkdir -p "$output_dir"

        echo "Processing sample: $sample_name"
        echo "Fastq1: $fastq1"
        echo "Fastq2: $fastq2"
        echo "Output: $output_dir"

        # Run the pipeline
        "$rowan_hic" run-hic \
          --fastq1 "$fastq1" \
          --fastq2 "$fastq2" \
          --genome-idx "$genome_idx" \
          --chrom-sizes "$chrom_sizes" \
          --assembly hg38 \
          --threads 55 \
          --output-dir "$output_dir"

        echo "Finished $sample_name"
        echo "--------------------------------------"
    fi
done
