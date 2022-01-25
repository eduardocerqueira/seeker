#date: 2022-01-25T16:56:37Z
#url: https://api.github.com/gists/d8a06af7a865713ffbfa8273dc987153
#owner: https://api.github.com/users/VincentRouvreau

import gudhi as gd
import numpy as np
import timeit
import os

pt = gd.read_points_from_off_file('SO3_50000.off')
np_pt = np.array(pt)
np.save("SO3_50000.npy", np_pt)

print ("SO3_50000.npy size = ", os.path.getsize("SO3_50000.npy"))
print("Uncompressed with numpy")
print("load(uncompressed) + save(uncompressed) + rm: ", timeit.timeit('import numpy as np; pt = np.load("SO3_50000.npy",mmap_mode="r"); np.save("test.npy", pt); import os; os.remove("test.npy")', number=100))
print("load(uncompressed) + sum:                     ", timeit.timeit('import numpy as np; pt = np.load("SO3_50000.npy",mmap_mode="r");pt.sum()', number=100))

np.savez_compressed("SO3_50000.npz", points=np_pt)
print ("SO3_50000.npz size = ", os.path.getsize("SO3_50000.npz"))
print("Compressed with numpy")
print("load(uncompressed) + save(compressed) + rm:   ", timeit.timeit('import numpy as np; pt = np.load("SO3_50000.npy",mmap_mode="r"); np.savez_compressed("test.npz", points = pt); import os; os.remove("test.npz")', number=100))
print("load(compressed) + sum:                       ", timeit.timeit('import numpy as np; npz = np.load("SO3_50000.npz",mmap_mode="r");npz["points"].sum()', number=100))

# SO3_50000.npy size =  3600128
# Uncompressed with numpy
# load(uncompressed) + save(uncompressed) + rm:  0.15597487400009413
# load(uncompressed) + sum:                      0.033888265999848954
# SO3_50000.npz size =  847256
# Compressed with numpy
# load(uncompressed) + save(compressed) + rm:    4.4320603249998385
# load(compressed) + sum:                        0.8238296199997421