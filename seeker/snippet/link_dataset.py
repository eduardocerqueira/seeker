#date: 2025-01-06T16:44:02Z
#url: https://api.github.com/gists/199a9771d3f50750d67a07d9b51002ce
#owner: https://api.github.com/users/aelkiss

from pairtree import id_encode
import sys
import os.path

def topath(ident):
    encodedid = id_encode(ident)
    filepath = []
    while encodedid:
        filepath.append(encodedid[:2])
        encodedid = encodedid[2:]
    return os.path.join(*filepath)

for line in sys.stdin:
    (namespace,objid) = line.strip().split('.',1)
    pt_objid = id_encode(objid)
    path = f"obj/{namespace}/pairtree_root/{topath(objid)}"

    print(f"mkdir -p $DATASET_HOME/{path}")
    print(f"ln -s /sdr1/{path}/{pt_objid} $DATASET_HOME/{path}/{pt_objid}")