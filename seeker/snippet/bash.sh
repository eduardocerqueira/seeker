#date: 2021-11-18T17:10:29Z
#url: https://api.github.com/gists/ff66ae7b522510c10efa0c68a92d7f9f
#owner: https://api.github.com/users/opinionsd


## using Ghostscript, poppler-utils & qpdf

# gs docs: https://ghostscript.com/doc/9.18/Use.htm#Other_parameters
# poppler: https://freedesktop.org/wiki/Software/poppler/
# qpdf docs: http://qpdf.sourceforge.net/files/qpdf-manual.html

# optimize pdf to use use CropBox, remove dublicate image refs, compress
gs -sDEVICE=pdfwrite -dUseCropBox -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -dDetectDuplicateImages=true -sOutputFile=test-out.pdf test.pdf

## could use "-dFastWebView=true" but it seems to break some PDFs

# qpdf: linearize the PDF (Fast Web View) 
qpdf --linearize test-out.pdf test-out-linearized.pdf

# review converted result
pdfinfo test-out.pdf

# review all images in pdf
pdfimages -list test-out.pdf

# validate
qpdf --check test-linearized.pdf