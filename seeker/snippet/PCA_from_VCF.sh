#date: 2022-11-16T16:55:52Z
#url: https://api.github.com/gists/55a2fdb5de4126c7a10caac0a16705fd
#owner: https://api.github.com/users/GabrielHoffman

ml plink/1.90b6.21 bcftools

FILE1=/sc/arion/projects/psychAD/data/share/cmu/MSSM-Penn-Pitt/CMC_OmniExpressExome_ImputationHRC_chr9.dose.vcf.gz
FILE2=/sc/arion/projects/psychAD/data/share/cmu/MSSM-Penn-Pitt/CMC_OmniExpressExome_ImputationHRC_chr19.dose.vcf.gz

# combine 2 VCFs into 1 file (just for faster testing)
bcftools concat $FILE1 $FILE2 | bgzip > combined.vcf.gz

# Concatenate All files.  (slower)
# bcftools concat /sc/arion/projects/psychAD/data/share/cmu/MSSM-Penn-Pitt/CMC_OmniExpressExome_ImputationHRC_chr*.dose.vcf.gz | bgzip > combined.vcf.gz

# tabix
tabix -fp vcf combined.vcf.gz

# Use plink to filder SNPs
# Get pruned SNPs for pca
plink --vcf combined.vcf.gz --double-id --maf 0.05 --indep-pairwise 1000 5000 0.2 

# Run pca on list of pruned SNPs
plink --vcf combined.vcf.gz --double-id --extract plink.prune.in --pca

# PCA stored here: plink.eigenvec

