#date: 2024-05-28T16:51:56Z
#url: https://api.github.com/gists/28b99bc3e7b0251a6e700fd8fe97ce98
#owner: https://api.github.com/users/BLINMATIC

import os
import zipfile
from PIL import Image

def create_folders():
    try: os.mkdir("IN"); print("1/2 IN")
    except: print("1/2 PASS IN")
    try: os.mkdir("OUT"); print("2/2 OUT")
    except: print("2/2 PASS OUT")

def create_cbz():
    for i in os.listdir("IN"):
        with zipfile.ZipFile("OUT/" + i + ".cbz", "w") as arch:
            for j in os.listdir(os.path.join("IN", i)):
                print("ADDING IMAGE " + os.path.join("IN", i, j))
                arch.write(os.path.join("IN", i, j), j)
            print("SAVING TO OUT/" + i + ".cbz")

def create_pdf():
    for i in os.listdir("IN"):
        images = []
        for j in os.listdir(os.path.join("IN", i)):
            if j.endswith(".jpeg") or j.endswith(".png") or j.endswith(".jpg"):
                print("ADDING IMAGE " + os.path.join("IN", i, j))
                images.append(Image.open(os.path.join("IN", i, j)))

        print("SAVING TO OUT/" + i + ".pdf")
        images[0].save("OUT/" + i + ".pdf", "PDF", resolution=100.0, save_all=True, append_images=images[1:])

print("CREATING FOLDERS")
create_folders()
print("CREATED FOLDERS")
print("================================================================")
print("MANGA PACKER BY BLINMATIC")
print("Put your manga chapter folders containing image files to the IN folder.")
print("Convert to .cbz [C] or .pdf [P]?")
if input("[C/P] > ") == "C":
    create_cbz()
else:
    create_pdf()
print("DONE")