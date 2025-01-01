#date: 2025-01-01T16:29:09Z
#url: https://api.github.com/gists/c329cc59d9822c5c73a6aa95e7531ec4
#owner: https://api.github.com/users/aelsi2

#!/usr/bin/env python3

# Copyright 2025 Andrey Eliseev

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pathlib as pl
import tempfile
import sys
import shutil
import argparse
 
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, StreamObject, DictionaryObject, IndirectObject, ArrayObject

TO_UNICODE_BYTES = b'''/CIDInit /ProcSet findresource begin
b'12 dict begin'
begincmap
/CIDSystemInfo
<<  /Registry (Adobe)
/Ordering (UCS)
/Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
40 beginbfchar
<0082> <201A>
<0084> <201E>
<0085> <2026>
<0086> <2020>
<0087> <2021>
<0088> <20AC>
<0091> <2018>
<0092> <2019>
<0093> <201C>
<0094> <201D>
<0095> <2022>
<0096> <2013>
<0097> <2014>
<00A0> <00A0>
<00A5> <0490>
<00A6> <00A6>
<00A7> <00A7>
<00A8> <0401>
<00A9> <00A9>
<00AA> <0404>
<00AB> <00AB>
<00AC> <00AC>
<00AD> <00AD>
<00AE> <00AE>
<00AF> <0407>
<00B0> <00B0>
<00B1> <00B1>
<00B2> <0406>
<00B3> <0456>
<00B4> <0491>
<00B5> <00B5>
<00B6> <00B6>
<00B7> <00B7>
<00B8> <0451>
<00B9> <2116>
<00BA> <0454>
<00BB> <00BB>
<00BC> <0458>
<00BD> <0405>
<00BF> <0457>
endbfchar
2 beginbfrange
<0020> <007E> <0020>
<00C0> <00FF> <0410>
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end'''

NAME_TYPE = NameObject("/Type")
NAME_FONT = NameObject("/Font")
NAME_TO_UNICODE = NameObject("/ToUnicode")

def fix_fonts(root, visited = set()):
    if isinstance(root, DictionaryObject):
        if NAME_TYPE in root and root[NAME_TYPE] == NAME_FONT:
            fix_font_if_needed(root)
            return
        for key, value in root.items():
            fix_fonts(value, visited)
    if isinstance(root, ArrayObject):
        for item in root:
            fix_fonts(item, visited)
    if isinstance(root, IndirectObject) and root not in visited:
        visited.add(root)
        fix_fonts(root.get_object(), visited)

def fix_font_if_needed(font):
    if NAME_TO_UNICODE in font:
        return
    to_unicode_object = StreamObject()
    to_unicode_object.set_data(TO_UNICODE_BYTES)
    font[NAME_TO_UNICODE] = to_unicode_object

def main():
    arg_parser = argparse.ArgumentParser(
            prog=sys.argv[0],
            description="Script for unshittifying Vaskevich's shitty PDFs.")
    arg_parser.add_argument("input", 
                            help="name of the shitty PDF")
    arg_parser.add_argument("-o", "--output", nargs=1, 
                            help="name of the unshittified PDF")
    args = arg_parser.parse_args()

    input_file_name = args.input
    input_file_path = pl.Path(args.input)
    output_file_name = args.output[0] \
        if args.output is not None \
        else str(input_file_path.with_stem(input_file_path.stem + "_unshit"))

    _, temp_file_name = tempfile.mkstemp()
    with PdfWriter(temp_file_name) as out_pdf:
        with PdfReader(input_file_name) as in_pdf:
            fix_fonts(in_pdf.root_object)
            for page in in_pdf.pages:
                out_pdf.add_page(page)
    shutil.copy(temp_file_name, output_file_name)

if __name__ == "__main__":
    main()
