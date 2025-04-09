#date: 2025-04-09T17:07:20Z
#url: https://api.github.com/gists/0a8dd8d944333afc7b65cca3efe58021
#owner: https://api.github.com/users/pashu123

def generate_mlir(m, n, k):
    # Define the MLIR types
    matA_type = f"tensor<{m}x{k}xf16>"
    matB_type = f"tensor<{n}x{k}xf16>"
    matCF32_type = f"tensor<{m}x{n}xf32>"

    file_name = f"file_{m}_{n}_{k}.mlir"

    # Generate the MLIR function
    mlir_code = f"""
    func.func @_{m}_{n}_{k}(%arg0: {matA_type}, %arg1: {matB_type}) -> {matCF32_type} {{
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %5 = tensor.empty() : {matCF32_type}
      %6 = linalg.fill ins(%cst : f32) outs(%5 : {matCF32_type}) -> {matCF32_type}
      %7 = linalg.matmul_transpose_b ins(%arg0, %arg1 : {matA_type}, {matB_type}) outs(%6 : {matCF32_type}) -> {matCF32_type}
      return %7 : {matCF32_type}
    }}
    """
    with open(file_name, "w") as file:
        file.write(mlir_code)


    return file_name

# Define the shapes
shapes = [
    (4, 14336, 4096),
    (4, 4096, 14336),
    (4, 128256, 4096),
    (4, 4096, 4096),
    (4, 1024, 4096)
]

def generate_compile_command(file_name, gen_x=True):
    compile_command = f'''~/iree-build/tools/iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" --iree-hal-dump-executable-files-to=dump/ --iree-dispatch-creation-enable-aggressive-fusion --iree-dispatch-creation-enable-fuse-horizontal-contractions=false --iree-opt-aggressively-propagate-transposes=true --iree-codegen-llvmgpu-use-vector-distribution=true --iree-opt-data-tiling=false --iree-vm-target-truncate-unsupported-floats --iree-opt-outer-dim-concat=true --iree-codegen-gpu-native-math-precision=true --iree-hal-indirect-command-buffers=true --iree-stream-resource-memory-model=discrete --iree-hal-memoization=true --iree-opt-strip-assertions --iree-global-opt-propagate-transposes=true --iree-opt-const-eval=false --iree-llvmgpu-enable-prefetch=true --iree-execution-model=async-external -iree-codegen-llvmgpu-test-vector-distribution-on-reduction {file_name} -o {file_name}_w.vmfb'''

    if(gen_x):
        compile_command = f'''~/iree-build/tools/iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))" --iree-hal-dump-executable-files-to=dump/ --iree-dispatch-creation-enable-aggressive-fusion --iree-dispatch-creation-enable-fuse-horizontal-contractions=false --iree-opt-aggressively-propagate-transposes=true --iree-codegen-llvmgpu-use-vector-distribution=true --iree-opt-data-tiling=false --iree-vm-target-truncate-unsupported-floats --iree-opt-outer-dim-concat=true --iree-codegen-gpu-native-math-precision=true --iree-hal-indirect-command-buffers=true --iree-stream-resource-memory-model=discrete --iree-hal-memoization=true --iree-opt-strip-assertions --iree-global-opt-propagate-transposes=true --iree-opt-const-eval=false --iree-llvmgpu-enable-prefetch=true --iree-execution-model=async-external {file_name} -o {file_name}.vmfb'''

    return compile_command

def benchmark_command(vmfb_file, m, n, k):
    benchmark_command = f'''iree-benchmark-module \
      --device=hip://0 \
      --device_allocator=caching \
      --module={vmfb_file} \
      --function=_{m}_{n}_{k} \
      --input={m}x{k}xf16=0.5 \
      --input={n}x{k}xf16=0.5 '''

    return benchmark_command


import os

# Generate MLIR code for each shape
for i, (m, n, k) in enumerate(shapes):
    file_name = generate_mlir(m, n, k)
    print("Running with vector distribution")
    compile_command = generate_compile_command(file_name, gen_x=False)
    os.system(compile_command)
    bench_cmd = benchmark_command(f"{file_name}_w.vmfb", m, n, k)
    print(bench_cmd)
    os.system(bench_cmd)
    print("Running without vector distribution")
    compile_command = generate_compile_command(file_name, gen_x=True)
    os.system(compile_command)
    bench_cmd = benchmark_command(f"{file_name}.vmfb", m, n, k)
    os.system(bench_cmd)
