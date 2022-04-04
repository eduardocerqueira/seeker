#date: 2022-04-04T16:58:20Z
#url: https://api.github.com/gists/f3230340c7c109127dedcc8300408451
#owner: https://api.github.com/users/u8sand

import json
import pandas as pd

df = pd.read_csv('https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=gd_status&col=gd_prev_sym&col=gd_aliases&col=gd_pub_chrom_map&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&col=gd_pub_eg_id&col=gd_pub_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit', sep='\t')

print('@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .')
print('@prefix gene_symbol: <https://identifiers.org/hgnc.symbol/> .')
print('@prefix semantic: <https://semanticscience.org/resource/> .')
print('')
print('semantic:SIO_001383 rdfs:comment "Gene Symbol"')
for _, row in df.iterrows():
  print(f"gene_symbol:{row['Approved symbol']} a semantic:SIO_001383 ;")
  print(f"  rdfs:label {json.dumps(row['Approved symbol'])} ;")
  print(f"  rdfs:comment {json.dumps(row['Approved name'])} .")
