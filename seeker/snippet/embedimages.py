#date: 2023-07-06T16:55:51Z
#url: https://api.github.com/gists/6e10aaf77cb8a699b2bbb8850bb9788e
#owner: https://api.github.com/users/bpanthi977

#!/usr/bin/env python
import sys
import os
import shutil 
import urllib.parse
import base64
import numpy as np
import csv
from PIL import Image
import time
import subprocess
import platform

class osDriver():
    def isdir(self, dir):
       return os.path.isdir(dir)

    def listdir(self, dir):
       return os.listdir(dir)

    def join(self, dir, basename):
       return os.path.join(dir, basename)

    def isfile(self, path):
       return os.path.isfile(path)

    def getsize(self, filepath):
       return os.path.getsize(filepath)

    def dirname(self, path):
       return os.path.dirname(path)

    def makedirs(self, dirpath):
       return os.makedirs(dirpath)

    def modified_time(self, path):
       return int(os.lstat(path).st_mtime) # in seconds

    def is_image(self, path):
        extensions3 = ['.png', '.jpg']
        extensions4 = ['.jpeg', '.heif']
        return path[-4:] in extensions3 or path[-5:] in extensions4
    
    def copyfile(self, src_driver, source, destination):
        if (isinstance(src_driver, osDriver)):
            return shutil.copyfile(source, destination)
        elif (isinstance(src_driver, ftpDriver)):
            src = src_driver.open(source, mode='rb')
            dest = open(destination, 'wb')
            src_driver.copyfileobj(src,dest)
            dest.close()
            src.close()


def load_metadata_cache(driver, dir):
    filename = driver.join(dir, 'embeddings.meta')
    if driver.isfile(filename):
       with open(filename, 'r') as f:
           reader = csv.reader(f)
           return [l for l in reader]
    else:
        return []

def save_metadata_cache(driver, dir, data):
    filename = driver.join(dir, 'embeddings.meta')
    with open(filename, 'w') as f:
        f.truncate()
        writer = csv.writer(f)
        writer.writerows(data)
        
def is_ctime_old(driver, ctime, filename):
    "ctime is the time the embeddings were created"
    return ctime + 10 < driver.modified_time(filename)
    
def process_recursively(driver, dir):
    if not driver.isdir(dir):
        return
    
    file_list = load_metadata_cache(driver, dir)
    basenames = [f[0] for f in file_list]
    # Get list of files
    for basename in driver.listdir(dir):
        filename = driver.join(dir, basename)
        
        if basename in basenames:
            idx = basenames.index(basename)
            ctime = int(file_list[idx][2])

            if is_ctime_old(driver, ctime, filename):
                del basenames[idx]
                del file_list[idx]
            else:
                continue
            
        if driver.isfile(filename):
            try:
                img_emb, desc_emb = get_embedding(driver, filename)
                file_list.append([basename, 
                                  driver.getsize(filename),
                                  driver.modified_time(filename),
                                  img_emb,
                                  desc_emb])
            except Exception as ex:
                print("[ERROR] Coundn't process file: " + filename + "\n\t", end='')
                print(ex)
                
        else :
            print('> ' + basename)
            for name, size, rPath, img_emb, desc_emb in process_recursively(driver, filename):
                file_list.append([basename + '/' + name, size, rPath, img_emb, desc_emb])

    save_metadata_cache(driver, dir, file_list)
    return file_list

EXIF_USER_COMMENT = 37510
EXIF_IMAGE_DESCRIPTION = 270
model = False

def load_model():
    global model
    if model:
        return model

    print('Loading clip-ViT-B-32 model ...')
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('clip-ViT-B-32')
    print('Model Loaded.')
    return model 

def decode(base64_str):
    return np.frombuffer(base64.decodebytes(bytes(base64_str, 'ascii')), dtype=np.float32)

def encode(np_arr):
    return ''.join(base64.encodebytes(np_arr.tobytes()).decode('ascii').splitlines())

 "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"( "**********"s "**********"t "**********"r "**********"i "**********"n "**********"g "**********") "**********": "**********"
    tokenizer = "**********"
    tokens = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"  "**********"< "**********"= "**********"  "**********"7 "**********"7 "**********": "**********"
        return [string]
    
    substrings = []
    tokens = tokens[1: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"r "**********"a "**********"n "**********"g "**********"e "**********"( "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********"/ "**********"/ "**********"6 "**********"0 "**********") "**********": "**********"
           substrings.append(tokenizer.decode(tokens[60*i: "**********"

    return substrings

def get_embedding(driver, path):
    if not driver.is_image(path):
        return '', ''

    image = Image.open(path)
    exif = image.getexif()
    if EXIF_USER_COMMENT in exif:
        try:
            data = exif[EXIF_USER_COMMENT].splitlines()[-3:]
            if len(data) == 3:
                ctime = int(data[0])
                if not is_ctime_old(driver, ctime, path):
                    img_emb = decode(data[1])
                    text_emb = decode(data[2])

                    assert img_emb.shape == (512,)
                    assert text_emb.shape == (512,)
                    return data[1], data[2]
        except:
            pass

    model = load_model()
    print('Processing ' + path + ' ... ', end='')
    sys.stdout.flush()
    
    img_emb = model.encode(image)
    if (EXIF_IMAGE_DESCRIPTION in exif):
        text_emb = "**********"=0)
    else:
        text_emb = np.zeros(512, dtype=np.float32)

    ctime = str(int(time.time()))
    img_emb = encode(img_emb)
    text_emb = encode(text_emb)
    
    oldcomment = exif[EXIF_USER_COMMENT] + '\n' if EXIF_USER_COMMENT in exif else ''
    exif[EXIF_USER_COMMENT] =  ctime + '\n' + img_emb + '\n' + text_emb
    print('[Saved]')

    if image.mode in ['RGBA', 'P'] and (path[-4:] == '.jpg' or path[-5:] == '.jpeg'):
        # file has jpeg or jpg extension but has a alpha channel 
        # remove the transparency layer 
        print("[WARNING] Image has transparency layer but is named as jpg or jpeg.")
        print("   Saving a backup copy and modifying the original: " + path)
        image.save(path+'.bak.png', exif=exif)
        image = image.convert("RGB")

    image.save(path, exif=exif)
    
    return img_emb, text_emb

def print_n(entries, count):
    if count:
        entries = entries[-count:]

    n = len(entries)
    for (score, name) in entries:
        print("{:4d} {:.6f} {:s}".format(n, score, name))
        n = n - 1


def search(driver, dir, query, count=False):
    data = load_metadata_cache(driver, dir)
    if len(data) == 0:
        print("No Metadata found. First run the command without query to compute embeddings")
        return

    model = load_model()
    query_emb = "**********"=0)
    q_norm = query_emb @ query_emb
    scores = []

    for (basename, size, mtime, img_enc, text_enc) in data:
        enc = decode(img_enc) + decode(text_enc)
        if enc.shape == (0, ):
            continue
        cosine_score = enc @ query_emb / ((enc @ enc) * q_norm)
        scores.append((cosine_score, basename))
    
    sorted_scores = sorted(scores, key=lambda s: s[0], reverse=False)
    print_n(sorted_scores, count)

    return sorted_scores

def repl_input():
    return input('> ').strip()
    
def open_img(path):
    if platform.system() == 'Darwin':       # macOS
        subprocess.run(['open', path], check=False)
    elif platform.system() == 'Windows':    # Windows
        os.startfile(path)
    else:                                   # linux variants
        subprocess.run(['xdg-open', path], check=False)

def parse_int(string):
    try:
        return int(string)
    except:
        return False    

def repl():
    print("EmbedImage REPL")
    print("Available commands are: index, search, cd, help, quit")

    driver = osDriver()
    entries = []
    while True:
        try: 
            inp = repl_input()
            if inp == 'quit' or inp == 'q':
                break
            elif inp == 'help' or inp == 'h':
                print("Available commands are: index, search, cd, help, quit")
                print("index             : Index images in the current director")
                print("search [...query] : Run a query against the indexed images")
                print(". [n]             : List n more search results from previous search")
                print("cd [path]         : change the directory")
                print("open [number]     : opens the file in the default image viewer")
            elif inp == 'cd':
                print(os.path.realpath(os.path.curdir))
            elif inp[:2] == 'cd':
                path = inp[3:]
                os.chdir(path)
                print(os.path.realpath(os.path.curdir))
            elif inp == 'index' or inp == 'i':
                print("Processing all files ...")                
                process_recursively(driver, './')
                print("Done.")
            elif inp == 'search':
                print("You need to enter the search terms")
            elif inp[0:6] == 'search':
                query = inp[7:]
                entries = search(driver, './', query, 10)
            elif inp[0:1] == 's':
                query = inp[2:]
                entries = search(driver, './', query, 10)
            elif inp[0:1] == '.':
                count = int(inp[2:])
                print_n(entries, count)
            elif inp[0:4] == 'open':
                idx = int(inp[5:])
                open_img(entries[-idx][1])
            elif parse_int(inp):
                idx = parse_int(inp)
                open_img(entries[-idx][1])

        except EOFError:
            break;
        except Exception as ex:
            print("[Error]", ex)
    

    
def cli():
    global files
    srcpath = os.path.abspath(sys.argv[1])

    if len(sys.argv) == 2:
        print("Processing all files ...")
        process_recursively(osDriver(), srcpath)
    else:
        query = ' '.join(sys.argv[3:])
        search(osDriver(), srcpath, query)
    
    print("Done.")

if len(sys.argv) == 2 and sys.argv[1] == "--help":
    print("""embedimages.py  Utility to store embedding of image into Exif metadata 
USAGE: embedimages.py srcdir [...query]
srcdir   - directory for whose images embedding will be computed (also subdirectories are traversed) 
query    - description to search the images with 
    
--help   - Shows this help
Bibek Panthi (bpanthi977@gmail.com)
""")
elif len(sys.argv) <= 1:
    repl()
else:
    cli()
    
