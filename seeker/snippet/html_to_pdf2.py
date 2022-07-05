#date: 2022-07-05T16:50:52Z
#url: https://api.github.com/gists/6544ac34aff7bd2de69e384e7101df59
#owner: https://api.github.com/users/misha-pyshark

import pdfkit

#Define path to wkhtmltopdf.exe
path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

#Define url
url = 'https://wkhtmltopdf.org/'

#Point pdfkit configuration to wkhtmltopdf.exe
config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

#Convert Webpage to PDF
pdfkit.from_url(url, output_path='webpage.pdf', configuration=config)