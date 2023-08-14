#date: 2023-08-14T16:53:39Z
#url: https://api.github.com/gists/f1e92d421312b7c7c1907667f4f3a318
#owner: https://api.github.com/users/wecacuee

#!/usr/bin/env python
try:
    import fitz
except ImportError as e:
    import subprocess
    print('Trying to install pip install PyMuPDF')
    subprocess.call("pip install PyMuPDF".split())
    print('Try pip install PyMuPDF')
    raise
import sys
if len(sys.argv) < 2:
    raise ValueError("Please provide a pdf to anonymize")
if len(sys.argv) < 3:
    outfilename = filename.replace('.pdf', '.anon.pdf')
else:
    outfilename = sys.argv[2]
filename = sys.argv[1]
doc = fitz.open(filename)
metadata = doc.metadata
for k, v in metadata.items():
    if k not in ['format']: # retain some metadata
        metadata[k] = ''
doc.set_metadata(metadata)

for page in doc:
    for annot in page.annots():
        info = annot.info
        info['title'] = ''
        annot.set_info(info)
        annot.update()

doc.save(outfilename)