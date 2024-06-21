#date: 2024-06-21T17:08:06Z
#url: https://api.github.com/gists/cf0d5cf93e36451e9d061cb292f21965
#owner: https://api.github.com/users/akornilotrust

#!/usr/bin/env python
# coding: utf-8

# In[56]:


from collections import Counter
from huggingface_hub import hf_hub_download
import json
import pandas as pd
import requests
import matplotlib.pyplot as plt


# In[2]:


# Get 1000 most downloaded models
hf_resp = requests.get("https://huggingface.co/api/models", params={"sort": "downloads", "limit": 1000})


# In[3]:


model_data = json.loads(hf_resp.text)
model_data[:1]


# In[6]:


# Note: Most Models that count not be fetch have restricted access and are not open-source

model_details = []

for model in model_data:
    hf_resp = requests.get(f"https://huggingface.co/api/models/{model['id']}")
    if hf_resp.status_code == 200:
        model_details.append(json.loads(hf_resp.text))
    else:
        print("Bad Response", model['id'])


# In[17]:


model_df = pd.DataFrame(model_details)


# In[ ]:





# In[24]:


# Count Licenses
tag_set =  Counter()
tag_col = []
open_c = 0
for model in model_details:
    had_license = False
    
    for tag in model['tags']:
        if tag.startswith('license'):
            tag_set[tag] += 1
            had_license = True
            tag_col.append(tag)
            break
    if not had_license:
        tag_set['no_license'] += 1
        tag_col.append('no_license')


# In[25]:


tag_set.most_common()


# In[38]:


common_license = [lic for lic, count in tag_set.items() if count >= 10]


# In[26]:


model_df['license'] = tag_col


# In[27]:


# First pull up explicit dataset links
has_dataset = []
for model in model_details:
    dataset_link = False
    if 'cardData' in model:
        if 'datasets' in model['cardData']:
            dataset_link = True
    has_dataset.append(dataset_link)


# In[28]:


model_df['dataset_link'] = has_dataset


# In[29]:


model_df['dataset_link'].mean()


# In[53]:


plt.style.use('ggplot')


# In[55]:


temp_df = model_df[model_df.license.isin(common_license)]
group_sum = temp_df.groupby('license').dataset_link.mean() * 100

group_sum.plot(kind='barh', title='Percent of Models with a Dataset Link by License')


# In[78]:


get_ipython().run_cell_magic('capture', '', '# Next grab the model\'s README to look for data references\n\nfnames = []\ntexts = []\n\nfor model in model_df.id.values:\n    try:\n        f = hf_hub_download(repo_id=model, filename="README.md")\n        fnames.append(f)\n        text = open(f).read()\n        texts.append(text)\n        \n    except KeyboardInterrupt:\n        break\n    except Exception as e:\n        print(e)\n        fnames.append(None)\n        texts.append("")\n')


# In[79]:


model_df["README"] = texts
model_df["cached_file_name"] = fnames


# In[83]:


for t in texts:
    if len(t) > 0 and '#' not in t:
        break


# In[ ]:





# In[93]:


model_df['data_section'] = model_df['README'].map(lambda x: '# Training Data' in x or '# Data' in x)

model_df['data_section'].mean()


# In[94]:


temp_df = model_df[model_df.license.isin(common_license)]

group_sum = temp_df.groupby('license').data_section.mean() * 100

group_sum.plot(kind='barh', title='Percent of Models with Data section in the README')


# In[88]:


model_df['data_mention'] = model_df['README'].map(lambda x: 'data' in x.lower())

model_df['data_mention'].mean()


# In[91]:


temp_df = model_df[model_df.license.isin(common_license)]

group_sum = temp_df.groupby('license').data_mention.mean() * 100

group_sum.plot(kind='barh', title='Percent of Models with a mention of Data in the README')


# In[ ]:





# In[96]:


# Sanity checks

model_df.groupby(['dataset_link', 'data_section', 'data_mention']).count()


# In[107]:


# Summary plot 1

group_sum = model_df[['data_mention', 'data_section', 'dataset_link']].mean() * 100
group_sum.index = ['Any Mention of Data', 'Data-related Markdown Section Title', 'Link to HF Dataset']
group_sum.plot(kind='barh', title='Analysis of Data Mentions in HF Model Cards')


# In[102]:


# Summary plot 2
temp_df = model_df[model_df.license.isin(common_license)]

group_sum = temp_df.groupby('license')[['data_mention', 'data_section', 'dataset_link']].mean() * 100
group_sum.columns = ['Any Mention of Data', 'Data Markdown Section', 'Link to HF Dataset']
group_sum.plot(kind='barh', title='Percent Data Mentions in Model Card by License Type', figsize=(20, 10))

