#date: 2022-06-16T16:54:05Z
#url: https://api.github.com/gists/44e6f9e21067d5efa3d7efe91472c28f
#owner: https://api.github.com/users/SoftSAR

from fpdf import FPDF

pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.add_page()
pdf.set_font("helvetica", "B", 16)
pdf.cell(40, 10, "Creating a PDF file using the FPDF2 library")
pdf.output("Example.pdf")
