#date: 2024-11-12T17:07:51Z
#url: https://api.github.com/gists/b7a5b3d0fe369d2f3d950f86b10ab8fc
#owner: https://api.github.com/users/sel-mort

#!/bin/env python3

# This script will "obfuscate" a python program by base64 encoding it's code and
#  then wrapping the result in a script that will base64 decode it...
#  ...THEN it gets the bytes of that new script and writes those out to a final script that
#  first evals the bytes to get the encoded script, then runs that script to run the real program.
# It's not very hard to un-obfuscate the generated python script, but it should be hard enough
#  the novice programmers will have a hard time doing it. Hopefully they'll find writing the code
#  themselves will be easier.
# You may be saying to yourself, "this is ridiculous" and I won't argue with that. It may be.
#  It's kinda fun though.

import base64
import sys
import os


def main(argv):
    input_filename = get_input_flename(argv)
    input_contents = read_input_file(input_filename)

    output_contents = obfuscate_base64(input_contents)
    output_contents = obfuscate_byte_list(output_contents)

    output_filname = get_output_filename(input_filename)
    write_output_file(output_filname, output_contents)


def get_input_flename(argv):
    if len(argv) < 2:
        die("Usage: obfuscator.py <file_to_obfuscate.py>")

    input_filename = argv[1]
    if not (input_filename.endswith(".py") and os.path.isfile(input_filename)):
        die(f"{input_filename} is not a python file")

    return input_filename


def read_input_file(input_filename):
    try:
        with open(input_filename) as input:
            return input.read()
    except:
        die(f"Error reading {input_filename}.")


def obfuscate_base64(contents):
    encoded = base64.b64encode(contents.encode())
    return f"""gobbledygook = {encoded}
import base64 as _
eval(compile(_.b64decode(gobbledygook),'<string>','exec'))"""


def obfuscate_byte_list(contents):
    byte_list = list(contents.encode())
    stringified_byte_list = ",\n    ".join(str(byte_list).split(","))
    return f"""def _(__,___):
    return (eval, (1-3//2) or ___(__({ stringified_byte_list }),'','exec'))
__ = _(bytes,compile)
__[0](__[1])"""


def get_output_filename(input_filename : str):
    without_ext = input_filename[:-3]
    return f"{without_ext}_obfuscated.py"


def write_output_file(output_filname, output_contents):
    try:
        with open(output_filname, "w") as output:
            output.write(output_contents)
    except:
        die(f"Error writing {output_filname}.")


def die(msg):
    print(msg)
    sys.exit(1)


if __name__ == '__main__':
    main(sys.argv)
