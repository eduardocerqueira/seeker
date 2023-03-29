#date: 2023-03-29T17:54:29Z
#url: https://api.github.com/gists/5bebf45f1ba4f8e2dfe89fa2b929b4f5
#owner: https://api.github.com/users/EdvardPotapenko

from PyPDF2 import PdfReader , PdfWriter
from os import walk

path = 'your_path_here'
angle_to_rotate = -90

filenames = next(walk('C:\\Users\\Mire\\Downloads\\rotated'), (None, None, []))[2]  # [] if no file

for filename in filenames:
    writer = PdfWriter()
    reader = PdfReader(path + filename)
    
    page = reader.pages[0]
    page.rotate(angle_to_rotate) 
    writer.add_page(page)
    output_file = open(path + filename, 'wb')
    writer.write(output_file)
    output_file.close()
    print(filename)

