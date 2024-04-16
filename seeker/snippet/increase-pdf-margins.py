#date: 2024-04-16T16:42:53Z
#url: https://api.github.com/gists/391ceb88a98ff73dacb1a4fb6ffaec1f
#owner: https://api.github.com/users/Splines

# pip install pymupdf

import fitz

src = fitz.open("QM_SS24_Skript.pdf")
doc = fitz.open()

LEFT_MARGIN = 100
RIGHT_MARGIN = 100

for page in src:
    width, height = page.rect.br
    placement = fitz.Rect(LEFT_MARGIN, 0, width + LEFT_MARGIN, height)

    newpage = doc.new_page(width=width + LEFT_MARGIN + RIGHT_MARGIN,
                           height=height)
    newpage.show_pdf_page(placement, src, page.number)

doc.save(f"cropped {src.name}")