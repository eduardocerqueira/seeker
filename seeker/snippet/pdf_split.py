#date: 2025-09-24T17:10:49Z
#url: https://api.github.com/gists/0f9edae94afa750f44e1fe4999783a7a
#owner: https://api.github.com/users/roslonek

from PyPDF2 import PdfFileWriter, PdfFileReader

# https://stackoverflow.com/questions/490195/split-a-multi-page-pdf-file-into-multiple-pdf-files-with-python#490203

def pdf_split(fname, start, end=None):
    print('pdf_split', fname, start, end)
    # pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 1 4

    #inputpdf = PdfFileReader(open("document.pdf", "rb"))
    inputpdf = PdfFileReader(open(fname, "rb"))
    output = PdfFileWriter()

    # turn 1,4 to 0,3
    num_pages = inputpdf.numPages
    if start:
        start-=1
    if not start:
        start=0
    if not end or end > num_pages:
        end=num_pages

    get_pages = list(range(start,end))
    #print('get_pages', get_pages, 'of', num_pages)
    # get_pages [0, 1, 2, 3]

    for i in range(start,end):
        if i < start:
            continue
        #output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))

    fname_no_pdf = fname
    if fname[:-4].lower() == '.pdf':
        fname_no_pdf = fname[:-4]
    out_filename = f"{fname_no_pdf}-{start+1}-{end}.pdf"
    with open(out_filename, "wb") as outputStream:
        output.write(outputStream)
    print('saved', out_filename)

'''
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',1,4)
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',5,8)
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',9,12)
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',13,16)
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',17,20)
pdf_split('~/Downloads/4-27-files/Invoice Email-0.pdf',21)
---
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 1 4
get_pages [0, 1, 2, 3] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-1-4.pdf
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 5 8
get_pages [4, 5, 6, 7] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-5-8.pdf
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 9 12
get_pages [8, 9, 10, 11] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-9-12.pdf
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 13 16
get_pages [12, 13, 14, 15] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-13-16.pdf
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 17 20
get_pages [16, 17, 18, 19] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-17-20.pdf
pdf_split ~/Downloads/4-27-files/Invoice Email-0.pdf 21 None
get_pages [20, 21, 22, 23] of 24
saved ~/Downloads/4-27-files/Invoice Email-0-21-24.pdf
'''