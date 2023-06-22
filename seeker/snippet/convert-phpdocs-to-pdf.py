#date: 2023-06-22T16:51:22Z
#url: https://api.github.com/gists/277b17c5fe4159d9e90177c3b4a43d22
#owner: https://api.github.com/users/gabrieleromanato

import tarfile
import os
import pdfkit

# Get the archive file at https://www.php.net/distributions/manual/php_manual_en.tar.gz
# Install pdfkit: https://github.com/JazzCore/python-pdfkit

def decompress_tar_file(file_name):
    tar = tarfile.open(file_name)
    tar.extractall()
    tar.close()

def get_html_files(dirpath):
    html_files = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            if file.endswith('.html'):
                html_files.append(os.path.join(root, file))
    return html_files

def sanitize_pdf_file_name(file_name):
    name_parts = file_name.split('.')
    parts = [n for n in name_parts if n != 'html']
    return '-'.join(parts) + '.pdf'

def convert_html_to_pdf(html_files):
    for file in html_files:
        parts = file.split('/')
        pdf_file = sanitize_pdf_file_name(parts[-1])

        with open(file, 'r') as f:
            pdfkit.from_file(f, f'./pdf/{pdf_file}', options={"enable-local-file-access": ""})
            print(f'Converted {file} to {pdf_file}')



def main():
    decompress_tar_file('php_manual_en.tar.gz')
    files = get_html_files('php-chunked-xhtml')
    convert_html_to_pdf(files)

if __name__ == '__main__':
    main()