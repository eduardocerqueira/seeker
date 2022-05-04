#date: 2022-05-04T17:07:36Z
#url: https://api.github.com/gists/0e03120151e7c799a5bd16d15fc3cfd5
#owner: https://api.github.com/users/watronfire

for SAMPLE in CTRL2-dil-500 CTRL48-dil-1000 CTRL2 CTRL23 CTRL14 CTRL48
do
	miniwdl run viral-pipelines/pipes/WDL/workflows/fastq_to_ubam.wdl \
    		FastqToUBAM.fastq_1=/home/training/laptop_setup-master/input/test_run/${SAMPLE}_spikein_R1.fastq.gz \
    		FastqToUBAM.fastq_2=/home/training/laptop_setup-master/input/test_run/${SAMPLE}_spikein_R2.fastq.gz \
    		FastqToUBAM.sample_name=${SAMPLE} \
    		FastqToUBAM.library_name=${SAMPLE} \
		FastqToUBAM.platform_name=ILLUMINA
done

for SAMPLE in CTRL2-dil-500 CTRL48-dil-1000 CTRL48 CTRL2 CTRL23 CTRL14 
do
	miniwdl run viral-pipelines/pipes/WDL/workflows/assemble_refbased.wdl \
		reads_unmapped_bams=/home/training/laptop_setup-master/output/test_run/bams/${SAMPLE}.unaligned.bam \
		reference_fasta=/home/training/laptop_setup-master/sars-cov-2_reference.fasta \
		trim_coords_bed=/home/training/laptop_setup-master/input/test_run/newprimers220408.bed \
		sample_name=${SAMPLE} \
		aligner=bwa
done


for SAMPLE in CTRL2-dil-500 CTRL48-dil-1000 CTRL48 CTRL2 CTRL23 CTRL14 
do
	miniwdl run viral-pipelines/pipes/WDL/workflows/align_and_count.wdl \
		align_and_count.reads_bam=/home/training/laptop_setup-master/output/test_run/bams/${SAMPLE}.unaligned.bam \
		align_and_count.ref_db=/home/training/laptop_setup-master/input/test_run/sdsi_spike-ins_2021-08-09_update.fasta
done

