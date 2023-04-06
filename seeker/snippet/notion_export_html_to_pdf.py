#date: 2023-04-06T17:02:20Z
#url: https://api.github.com/gists/633be046a0dd958fffe993de7935668e
#owner: https://api.github.com/users/albertopasqualetto

# Author: Alberto Pasqualetto
# This code is licensed under MIT license
#
# Create a pdf for each html file exported by Notion in the current directory
# and merge them in one pdf if required with -m or --merge
# if only the merge of files in './pdfs' is required use -M or --only-merge

import sys
from pathlib import Path
from weasyprint import HTML
from pypdf import PdfMerger

def convert():
	new_pdfs_list = []

	Path("./pdfs").mkdir(parents=True, exist_ok=True)

	# for each file in this directory
	for file in Path('.').glob('*.html'):
		# create a pdf with the same name
		new_pdf = Path('./pdfs/' + ' '.join(file.stem.split(' ')[:-1]) + '.pdf')
		HTML(file).write_pdf(new_pdf)
		new_pdfs_list.append(new_pdf)
	
	return new_pdfs_list

def merge_pdfs(new_pdfs_list):
	if len(new_pdfs_list) > 1:
		# merge all pdfs in one
		merger = PdfMerger()
		for pdf in new_pdfs_list:
			merger.append(pdf)
		merger.write(Path('./pdfs/' + Path('.').resolve().name + '_merged.pdf'))
		merger.close()


if '__main__' == __name__:
	# get is merge is required with the first argument
	merge = False
	if len(sys.argv) > 1:
		merge = sys.argv[1] == '-m' or sys.argv[1] == '--merge'
		only_merge = sys.argv[1] == '-M' or sys.argv[1] == '--only-merge'

	# convert html to pdf
	if not only_merge:
		new_pdfs_list = convert()

	# merge pdfs
	if merge:
		merge_pdfs(new_pdfs_list)

	if only_merge:
		merge_pdfs(['./'+str(p) for p in Path('./pdfs').glob('*.pdf')])
	
