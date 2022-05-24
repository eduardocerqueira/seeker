#date: 2022-05-24T17:07:29Z
#url: https://api.github.com/gists/78a3ffdd11f8180332002175203bf28f
#owner: https://api.github.com/users/garland3

from pathlib import Path
from fpdf import FPDF
import matplotlib.pyplot as plt

class MyPDF(FPDF):
        pass

pdf  = MyPDF(unit = 'pt', format = 'A4')
print("pdf size is ", pdf.w, pdf.h)
pdf.add_page()
plots  = list(Path("viz").glob("*.png"))
plots = [p1 for p1 in plots if str(p1).find("rescale")==-1]
print(plots)
i = 0
for plot in plots:
    im = Image.open(plot)
    print(im.size)
    t_size = 1000
    if im.size !=(t_size,t_size):
        print("rescale")
        im = im.resize((t_size,t_size))
        new_name = Path("viz") / f"rescaled{i}.png"
        im.save(new_name)
        plot = new_name
    pdf.image(str(plot), w = 500, h = 500)
    pdf.add_page()
    i +=1
pdf.output("viz/plots.pdf", "F")