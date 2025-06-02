#date: 2025-06-02T17:10:03Z
#url: https://api.github.com/gists/63a7d8efe08c2739916fb97e8113e85d
#owner: https://api.github.com/users/tomolopolis

from medcat.cat import CAT
from medcat.cdb import CDB
from typing import List
import json

cat = CAT.load_model_pack('<model pack zip>')

all_cuis = json.load(open('relevant_cuis.json'))

cat.config.linking['filters']['cuis'] = all_cuis

clinical_text = """
patient has  diabetes, hypertension, stroke 2 years ago and has increased central adeposity, non-small cell lung cancer

Patient has visited my clinic today and has issues related to a recent fall.

Current complaint: retinopathy, foot ulcer, and syncopy, currently prescribed Midodrine 2.5mg 3 times od.
"""

ents = cat.get_entities(clinical_text)['entities'].values()

# filter to those ents that have meta_anns that:
# Prescene: True
# Subject: Patient
# Time: Recent
filtered_ents = [e for e in ents if e['meta_anns']['Presence']['value'] == 'True' and e['meta_anns']['Subject']['value'] == 'Patient'
                 and e['meta_anns']['Time']['value'] == 'Recent']                     

filtered_ents