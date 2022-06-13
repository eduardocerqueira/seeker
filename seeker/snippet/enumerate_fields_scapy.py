#date: 2022-06-13T17:02:16Z
#url: https://api.github.com/gists/267534283414fe4fbbc837f495cbf497
#owner: https://api.github.com/users/lain3d

def genexprs(p, nlayer, all=False, pname='p'):
    s = ""
    for i,f in enumerate(p.getlayer(nlayer).fields):
        # print(f"class name: {p.getlayer(nlayer).__class__.__name__}]")
        val = getattr(p.getlayer(nlayer), f)
        
        if all or val != p.getlayer(nlayer).fields_desc[i].default:
            if isinstance(val, int):
                val = hex(val)
            elif isinstance(val, bytearray) or isinstance(val, bytes):
                val = f'bytes.fromhex("{val.hex()}")'
            # if matches default, dont bother
            s += ( f'{pname}[{p.getlayer(nlayer).__class__.__name__}].{f} = {val}\n')

    print(s)
    print("done")