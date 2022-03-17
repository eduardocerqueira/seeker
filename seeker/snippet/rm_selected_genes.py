#date: 2022-03-17T17:06:10Z
#url: https://api.github.com/gists/6015b7cf1e61e21ba945d81f3ab6cbac
#owner: https://api.github.com/users/silvtal

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:02:33 2022
@author: silvia
"""

# =============================================================================
# load modules, data and chose excluded genes
# =============================================================================
import pandas as pd
import os
import re

os.chdir("/home/silvia/AAA/2022-02-21_TnSeq_simul/Grinder-0.5.4")

in_file = "F113newannot_2020.csv" # sep = ","

tab_annot = pd.read_csv(in_file, sep=",", 
                        usecols=lambda x: x in ["length", "start", "end", "strand", "productfeat"])

# I want to exclude these genes (because they are essential, they won't have 
# inserted elements in them)

lst = ["Motility protein A",
       "Motility protein B",
       "Methyl-accepting chemotaxis protein CtpL",
       "Methyl-accepting chemotaxis protein McpS",
       "Methyl-accepting chemotaxis protein McpU",
       "Methyl-accepting chemotaxis protein McpP",
       "Methyl-accepting chemotaxis protein PctA",
       "Chemotaxis protein CheY",
       "Chemotaxis protein CheW"]

excluded = tab_annot.query('productfeat in @lst')

print("Mock genes to exclude")
print("---------------------")
print(excluded)

# =============================================================================
# exclude them from the whole genome .fna
# =============================================================================
f = open("F113.fna", "r")
fna = [line.strip() for line in f.readlines()]
f.close() 

header = fna[0]
genome = fna[1]

# get the sequences we want to remove
exclu_seqs = []
for index, row in excluded.iterrows():
    s_e = sorted([row["start"],row["end"]])
    exclu_seqs.append(genome[(s_e[0]+1):(s_e[1]+2)]) # +1 at the start 'cause python; extra +1 at the end so it includes the whole seq

# remove repeated sequences and exclude them
genomenew = re.sub(r'|'.join(map(re.escape, set(exclu_seqs))), '', genome)

print("length before excluding: "+str(len(genome)))
print("length AFTER excluding: "+str(len(genomenew)))

# save file
with open("filtered_F133.fna", "w") as f:
    f.write(header+"\n")
    f.write(genomenew)