#date: 2021-10-28T17:12:52Z
#url: https://api.github.com/gists/0651a6172916a88f461e722203030ed5
#owner: https://api.github.com/users/tpatton-cohere

import time
import re
from cohereML.s3_utils import getOCRText
from cohere_analytics.api_service import ApiService
import json

t0 = time.time()
guideline_ct_mapping = {}
text_list = []
CT_presence = 'conservative treatment|conservative management|exercise program|home exercise|physical therapy|non-surgical, non-injection care'
ocr = getOCRText('prod', config_okta.okta_username, config_okta.okta_password)
api = ApiService('prod', config_okta.okta_username, config_okta.okta_password)
sample_df = merged_df.iloc[:500]
sr_guideline_mapping = {}

for i in range(sample_df.shape[0]):
    if i % 500 == 0:
        print(i/sample_df.shape[0], time.time() - t0)
        
    sr_id = sample_df.iloc[i]['_id']
    guidelines = api.get_request(f'serviceRequestGuidelines?serviceRequestId={sr_id}')
    if len(guidelines) != 0:
        applicable_guidelines = []
        for i, guideline in enumerate(guidelines):
            guideline_ids = guidelines[i]['guidelineIds']
            applicable_guidelines.extend(guideline_ids)
            for j, guideline_id in enumerate(guideline_ids):
                if guideline_id not in guideline_ct_mapping.keys():
                    guideline_str = re.sub('<[^<]+?>', '', guidelines[i]['guidelines'][j]['guidelineHtml']).lower()
                    guideline_ct_mapping[guideline_id] = bool(re.search(CT_presence, guideline_str))
        sr_guideline_mapping[sr_id] = applicable_guidelines
        
    else:
        sr_guideline_mapping[sr_id] = None
    
print(time.time() - t0)
print(guideline_ct_mapping)
sr_guideline_mapping