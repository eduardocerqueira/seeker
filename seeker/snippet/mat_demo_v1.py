#date: 2024-08-30T16:54:09Z
#url: https://api.github.com/gists/128b947a4b64527edd4b7b64fbee926f
#owner: https://api.github.com/users/MicahGale

import montepy

mat = montepy.read_input(...).materials[5]

list(mat)
# would be
[
  (Isotope(...), 0.05),
  (Isotope(...), 0.10),
  ...
]

# would be a list
component = mat[1]

# searchable by find
mat.find("U-235m1.80c")
# returns
[
  (Isotope("U-235m1.80c"), 0.01),
  (Isotope("U-235m1.80c"), 0.5)
]

# searchable by element, A, library
mat.find(element=Element(92), A=235, library="80c")
#accepts slices
mat.find(element=slice(1,92))

#accepts functions
mat.find(element=lambda e: "ium" in e.name)
