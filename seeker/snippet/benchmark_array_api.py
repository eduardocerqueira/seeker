#date: 2026-02-09T17:25:11Z
#url: https://api.github.com/gists/fa89a941927676ba919289c2d3d05021
#owner: https://api.github.com/users/VincentRouvreau

from sklearn.datasets import fetch_openml
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from timeit import default_timer as timer
from datetime import timedelta

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

X = X.reshape((-1, 28, 28))

for n_jobs in [1, -2, 2]:
    print(f"* Scikit-learn {n_jobs=}")
    cp = CubicalPersistence(homology_dimensions=0, n_jobs=n_jobs)
    start = timer(); diags = cp.fit_transform(X); print(timedelta(seconds=timer()-start))

try:
    from gudhi.array_api import cubical_persistence
    import torch
    
    print("* Array-api NumPy")
    start = timer()
    diags = [cubical_persistence(cells, homology_dimensions=[0]) for cells in X]
    print(timedelta(seconds=timer()-start))
    
    X = torch.tensor(X)
    for preserve_gradient in [False, True]:
        print(f"* Array-api PyTorch {preserve_gradient=}")
        start = timer()
        diags = [cubical_persistence(cells, homology_dimensions=[0], preserve_gradient=preserve_gradient) for cells in X]
        print(timedelta(seconds=timer()-start))
except:
    pass

### master
# 
# * Scikit-learn n_jobs=1
# 0:00:01.053174
# * Scikit-learn n_jobs=-2
# 0:00:03.751556
# * Scikit-learn n_jobs=2
# 0:00:03.209264
# 
### array_api_compat
# 
# * Scikit-learn n_jobs=1
# 0:00:01.285406
# * Scikit-learn n_jobs=-2
# 0:00:04.444217
# * Scikit-learn n_jobs=2
# 0:00:03.824414
# * Array-api NumPy
# 0:00:01.196292
# * Array-api PyTorch preserve_gradient=False
# 0:00:02.167514
# * Array-api PyTorch preserve_gradient=True
# 0:01:09.260835