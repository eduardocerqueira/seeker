#date: 2022-10-11T17:09:52Z
#url: https://api.github.com/gists/80a2a0065d123cb2c10ee3842ff6cb91
#owner: https://api.github.com/users/7bitlyrus

import pdfrw # pip install pdfrw
import sys

if not len(sys.argv) != 1:
    print(f'usage: {sys.argv[0]} filename')
    exit()
    
filename = sys.argv[1]
template_pdf = pdfrw.PdfReader(filename)
for Page in template_pdf.pages:
	if Page['/Annots']:
		for annotation in Page['/Annots']:
			annotation.update(pdfrw.PdfDict(Ff=0))

if template_pdf.Root.AcroForm is not None:
	template_pdf.Root.AcroForm.update(pdfrw.PdfDict(NeedAppearances=pdfrw.PdfObject('true')))
else:
    print("form not found")
    exit()
    
pdfrw.PdfWriter().write('unflatten-' + filename, template_pdf)
print(f'Written to unflatten-{filename}')