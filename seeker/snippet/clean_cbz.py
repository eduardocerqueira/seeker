#date: 2023-01-31T16:59:55Z
#url: https://api.github.com/gists/70ac107af03867341da4667653e43917
#owner: https://api.github.com/users/corylk

import hashlib
import os
import pdb
import sys
import uuid
import zipfile
 
path = sys.argv[1]
threshold = int(sys.argv[2]) or 1
commit = '-c' in sys.argv

for zfile in os.listdir(path):
    if not zfile.endswith('.cbz'):
        continue

    hashes = {}
    fnames = {}
    tfile = f'/tmp/{uuid.uuid1()}/'
 
    with zipfile.ZipFile(zfile, 'r') as z:
        for fname in z.namelist():
            zinfo = z.getinfo(fname)

            if zinfo.is_dir():
                continue

            with z.open(fname, 'r') as f:
                data = f.read()
                m = hashlib.md5(data)
                h = m.hexdigest()
                if h not in hashes:
                    hashes[h] = 0
                    fnames[h] = []
                hashes[h] += 1
                fnames[h].append(fname)
    
    if not commit:
        with zipfile.ZipFile(zfile, 'r') as z:
            for (h, c) in hashes.items():
                if c > threshold:
                    print(h)
                    for fname in fnames[h]:
                        print(f'└ {fname}')
                        z.extract(fname, 'img/')
    else:
        with zipfile.ZipFile(zfile, 'r') as z:
            z.extractall(tfile)
    
        for (h, c) in hashes.items():
            if c > threshold:
                for fname in fnames[h]:
                    print(f'Removing {fname}')
                    os.remove(f'{tfile}{fname}')
    
        with zipfile.ZipFile(zfile, 'w') as z:
            for root, dirs, files in os.walk(tfile):
                for file in files:
                    z.write(os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), tfile))
