#date: 2023-12-20T16:50:53Z
#url: https://api.github.com/gists/75d93455a19f9c5dad32da9929ce6e76
#owner: https://api.github.com/users/zach-morris

# Creates a list of all supported softlist systems in MAME/MESS using them with python.  Quick and dirty
# As described here: https://forums.libretro.com/t/guide-play-non-arcade-systems-with-mame-or-mess/17728/
# Using re, requests, and pandas


import pandas as pd
import re, requests, xmltodict
from pathlib import Path

with requests.get('https://github.com/mamedev/mame/tree/master/hash') as url:
	hash_files = [x for x in re.findall('{"name":".*?","path":"hash/(.*?)","contentType":"file"}',url.text) if x.endswith('xml')]

softlists = list()

for hh in hash_files:
	with requests.get('https://raw.githubusercontent.com/mamedev/mame/master/hash/{}'.format(hh),headers={'Range':'bytes=0-8000'}) as url:
		print('Grabbing softlist info for {}'.format(hh))
		cc = re.findall('<softwarelist name="(.*?)" description="(.*?)">',url.text)
		if isinstance(cc,list) and len(cc)==1:
			softlists.append({'System':cc[0][-1],'Folder Name':cc[0][0],'Hash File':hh})

softlist_df = pd.DataFrame.from_dict(softlists)

bios_files = pd.DataFrame.from_dict(xmltodict.parse(Path('/path_to_progretto_dat_file/MAME 0.261 (mess).dat').read_text()).get('datafile').get('machine'))    
bios_files['softwarelist'].apply(lambda x: x.get('@name') if isinstance(x,dict) else None)
softlist_df = softlist_df.merge(bios_files.groupby('softwarelist_name').agg({'@name':list}).rename(columns={'@name':'BIOS'}).reset_index(),left_on='Folder Name',right_on='softwarelist_name')
softlist_df['BIOS Files'] = softlist_df['BIOS'].apply(lambda x: ','.join(sorted(set([y for y in x if isinstance(y,str)]))) if isinstance(x,list) else None)
ff=Path('mame_softlists.md')
ff.write_text(softlist_df[[x for x in softlist_df.columns if x not in ['BIOS','softwarelist_name']]].to_markdown(index=False))
