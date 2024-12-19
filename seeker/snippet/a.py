#date: 2024-12-19T16:56:50Z
#url: https://api.github.com/gists/06c23a25e37e69f3de05c9d031e1512f
#owner: https://api.github.com/users/vmiheer

#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import torch
import dgl.sparse as dglsp
from copy import deepcopy
from itertools import product, repeat
from more_itertools import take
import argparse
import sys


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def tensor_from_file(file_name: Path, dtype=np.float32, read_size=-1):
    if read_size > 0:
        a = open(file_name, "rb").read(read_size)
    else:
        a = open(file_name, "rb").read()
    n = np.frombuffer(a, dtype=dtype)
    return torch.from_numpy(deepcopy(n))


if is_interactive():
    sys.argv = [
        "a.py",
        "banded_11_r2.coordinates.bin",
        "11",
        "4",
        "2",
    ]
    # sys.argv = [
    #     "a.py",
    #     "banded_4_r2.coordinates.bin",
    #     "4",
    #     "1",
    #     "1",
    #     "/scratch/general/vast/u1290058/LAPIS_Workspace/snl-utah/sandboxes/mseyden/graph-attention/dgl/arxiv.coordinates.bin",
    #     "169344",
    #     "1",
    #     "1",
    # ]

if len(sys.argv) != 5:
    print(f"Usage: python3 {sys.argv[0]} <dataset_name> <N> <Dh> <Nh>")
    exit(1)

N, Dh, Nh = map(int, sys.argv[2:])

A = tensor_from_file(Path(sys.argv[1]), dtype=np.int64)
nnzCount = A.shape[0] // 2
info_file_name = sys.argv[1].split(".")[0] + ".coordinates.info"
Nv, _, Nnz = map(int, open(info_file_name).read().split())
print(f"Nv: {Nv}, N: {N}, Nnz: {Nnz}, nnzCount: {nnzCount}")
print(f"N: {N}, Dh: {Dh}, Nh: {Nh}")
# assert N == Nv
edgeFeatPath = sys.argv[1].split(".")[0] + ".edge.data.bin"
edgeData = tensor_from_file(edgeFeatPath, read_size=(Nnz * Nh * 4)).reshape(Nnz, Nh)
print("EdgeData[0]: ", edgeData[0])
A = dglsp.spmatrix(
    A.reshape(Nnz, 2).transpose(1, 0),
    # val=torch.tensor([1.0] * Nnz),
    val=edgeData,
    shape=(N, N),
)
featPath = sys.argv[1].split(".")[0] + ".vert.data.bin"
# Q = tensor_from_file(featPath, read_size=(N * Dh * Nh * 4)).reshape(N, Dh, Nh)
# K = tensor_from_file(featPath, read_size=(N * Dh * Nh * 4)).reshape(N, Dh, Nh)
V = tensor_from_file(featPath, read_size=(N * Dh * Nh * 4)).reshape(N, Dh, Nh)

spmm_out = dglsp.bspmm(A, V)
print(spmm_out.shape)
outPath = sys.argv[1].split(".")[0].split("/")[-1] + ".res"
out = tensor_from_file(outPath, read_size=(N * Dh * Nh * 4)).reshape(N, Dh, Nh)

# sddmm_out = dglsp.bsddmm(A, Q, K.transpose(1, 0)).softmax(dim=1)
# spmm_out = dglsp.bspmm(sddmm_out, V)
## sddmm_out = dglsp.bsddmm(A, Q, K.transpose(1, 0)).to_dense().softmax(dim=1)
## spmm_out = torch.mm(torch.squeeze(sddmm_out), torch.squeeze(V, -1))
## # print(spmm_out)
## sddmm_out = dglsp.bsddmm(A, Q, K.transpose(1, 0)).softmax(dim=1)
## # print(sddmm_out)
## spmm_out = dglsp.bspmm(sddmm_out, V)
## print(sddmm_out)
## exit(1)
## spmm_out = torch.mm(sddmm_out, V)

if not (torch.allclose(out, spmm_out)):
    print(spmm_out - out)
    print(spmm_out)
    print(out)
    exit(1)
exit(0)
