#date: 2023-06-09T17:06:51Z
#url: https://api.github.com/gists/80b86cc4a256db502b5d8bed3b857113
#owner: https://api.github.com/users/igoforth

# C:\"Program Files"\LibreOffice\program\soffice.exe --headless --convert-to docx --outdir out in\{filename}
# structure:
# .
# out\
# in\
# convert.py

import os
import subprocess

def convert(filename):
    print("Converting " + filename)
    subprocess.call(["C:\\Program Files\\LibreOffice\\program\\soffice.exe",
                     "--headless",
                     "--convert-to",
                     "docx",
                     "--outdir",
                     "out",
                     "in\\" + filename])

def main():
    if not os.path.exists("in"):
        print("No input directory")
        return
    if not os.path.exists("out"):
        os.mkdir("out")
    for filename in os.listdir("in"):
        if filename.endswith(".doc"):
            convert(filename)
        else:
            print("Skipping " + filename)

if __name__ == "__main__":
    main()