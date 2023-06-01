#date: 2023-06-01T16:51:17Z
#url: https://api.github.com/gists/7f6ac404e16df6e5f5ee69904a06e172
#owner: https://api.github.com/users/pashu123

from iree import runtime as ireert
from iree.compiler import compile_str

import numpy as np
import os

with open(os.path.join("vicuna_fp32_cpu.vmfb"), "rb") as mlir_file:
    flatbuffer_blob = mlir_file.read()


backend = "llvm-cpu"
args = ["--iree-llvmcpu-target-cpu-features=host"]

config = ireert.Config("local-task")
vm_module = ireert.VmModule.from_flatbuffer(config.vm_instance, flatbuffer_blob)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
complex_compiled = ctx.modules.module

input1 = np.load("inp1.npy")
input2 = np.load("inp2.npy")
x = complex_compiled.forward(input1, input2)

print(x.to_host())
