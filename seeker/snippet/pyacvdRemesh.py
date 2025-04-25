#date: 2025-04-25T16:44:41Z
#url: https://api.github.com/gists/e96019458432609e42ec6baf49f7abc9
#owner: https://api.github.com/users/robin-gdwl

import pyvista as pv
import pyacvd

mesh = pv.read('einkaufswagenMAX.obj')

mesh.plot(show_edges=True, color='w')
clus = pyacvd.Clustering(mesh)

# mesh is not dense enough for uniform remeshing
#clus.subdivide(3)
clus.cluster(12000)

# plot clustered mesh
clus.plot()

# remesh
remesh = clus.create_mesh()

# plot uniformly remeshed cow
remesh.plot(color='w', show_edges=True)

pl = pv.Plotter()
_ = pl.add_mesh(remesh)
pl.export_obj("test2.obj")