#date: 2022-07-05T16:49:37Z
#url: https://api.github.com/gists/2da2b8ff3f853dcafd25ce39d492e151
#owner: https://api.github.com/users/misha-pyshark

import pdfkit

#Define path to wkhtmltopdf.exe
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

#Define path to HTML file
path_to_file = 'sample.html'

#Point pdfkit configuration to wkhtmltopdf.exe
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

#Convert HTML file to PDF
pdfkit.from_file(path_to_file, output_path='sample.pdf', configuration=config)