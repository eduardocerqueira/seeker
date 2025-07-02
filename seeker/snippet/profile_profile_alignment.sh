#date: 2025-07-02T16:55:28Z
#url: https://api.github.com/gists/049024ce968a49d56d4e41b5c8dc0edb
#owner: https://api.github.com/users/PraljakReps

###################
# Align reference #
###################

# export a neighbour joining guide tree to the Newick format for a given reference dataset (e.g. SH3 natural sequences)
./famsa -gt nj -gt_export ./test/SH3_natural.fasta nj_SH3_natural.dnd

# align sequences with the previously generated guide tree
./famsa -gt import nj_SH3_natural.dnd ./test/SH3_natural.fasta nj_SH3_natural.aln

################
# Align target #
################

# export a neighbour joining guide tree to the Newick format for a given reference dataset (e.g. SH3 design sequences)
./famsa -gt nj -gt_export ./test/SH3_designs.fasta nj_SH3_designs.dnd

# align sequences with the previously generated guide tree
./famsa -gt import nj_SH3_designs.dnd ./test/SH3_designs.fasta nj_SH3_designs.aln

###################
# Trim alignments #
###################

# More aggressive gap removal
trimal -in nj_SH3_natural.aln \
       -out nj_SH3_natural_trimmed.aln \
       -gappyout
       
# More aggressive gap removal
trimal -in nj_SH3_designs.aln \
       -out nj_SH3_designs_trimmed.aln \
       -gappyout       
    
#############################
# profile-profile alignment #
#############################
   
# profile-profile alignment without refining output 
./famsa -refine_mode off \
        nj_SH3_natural_trimmed.aln \
        nj_SH3_designs_trimmed.aln \
        SH3_natural_vs_designs_pp.aln
 
 
# (optional) Extract aligned positions that correspond to desired WT sequence
# write a script to extract positions (e.g. extract_wt_positions.py) 
        
     
