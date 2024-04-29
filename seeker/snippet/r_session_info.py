#date: 2024-04-29T17:01:20Z
#url: https://api.github.com/gists/6543588b2a92582be1e65ac6a3d73568
#owner: https://api.github.com/users/dtenenba

"""
Given the lines in a sessionInfo() that start (minus whitespace) with `[`, 
this creates a dict of package name/version pairs.
"""

def get_pkgs(text):
    """
    Get a dict of package/version pairs from the packages section
    of R's sessionInfo() output.
    """
    out = {}
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line.startswith("["):
            continue
        segs = line.split()
        del segs[0]
        for seg in segs:
            pkg, ver = seg.rsplit("_", 1)
            out[pkg] = ver
    return out
  