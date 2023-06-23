#date: 2023-06-23T17:09:51Z
#url: https://api.github.com/gists/7d91387fdc28e8b5cc14794f500d98c6
#owner: https://api.github.com/users/mdvsh

import re, mmap
pattern = rb'pick_rsteps\s*:\s*(\d+)\s+place_rsteps\s*:\s*(\d+)'

expfnam = ["svfu", "svu", "mr"]
nf = 10
for exp in expfnam:
    means = []
    for i in range(nf):
        with open(f"dat/{exp}{i}.txt", "r") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mmap_file:
                match = re.search(pattern, mmap_file)
                if match:
                    p1, p2 = int(match.group(1)), int(match.group(2))
                    means.append((p1, p2))
    mean_tuple = tuple(map(lambda x: sum(x) / len(means), zip(*means)))
    print(f"mean {exp}: ", mean_tuple)