#date: 2025-07-18T17:03:44Z
#url: https://api.github.com/gists/264c68fcd9298568dc62fd57e80b749b
#owner: https://api.github.com/users/pashu-cohere

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from typing import Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

"""
A high-performance batched dense fused swiglu (C = sigmoid(A x W1)*(A x W2)) 
example for the NVIDIA Hopper architecture using CUTE DSL.

This GEMM kernel inside swiglu supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Supports multi-stage pipeline to overlap computation and memory access

.. code-block:: bash

    python examples/hopper/swiglu.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --w_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --w_major k --c_major n

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/swiglu.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --w_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --w_major k --c_major n

"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[(128, 128, 64), (128, 256, 64), (128, 64, 64), (64, 64, 64)],
        default=(128, 128, 64),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--w_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--w_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--tolerance", type=float, default=1e-03, help="Tolerance for validation"
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mnk) != 3:
        parser.error("--tile_shape_mnk must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////

@cute.jit
def sigmoid(x: cute.TensorSSA | cutlass.Float32 , y: cute.TensorSSA | cutlass.Float32) -> cute.TensorSSA:
    """Element‑wise 1 / (1 + exp(‑x)) for both cute.TensorSSA and scalar Float32."""
    one = cutlass.Float32(1.0)
    neg_one = cutlass.Float32(-1.0)

    # Tensor path – unroll at IR‑compile time
    tmp = cute.make_fragment(x.shape, cutlass.Float32)
    tmp.store(x)
    tmp2 = cute.make_fragment(x.shape, cutlass.Float32)
    tmp2.store(y)

    for i in cutlass.range_constexpr(cute.size(x.shape)):
        v = tmp[i]
        # exp is base‑e; if your build only exposes exp2, replace with
        #    cute.arch.exp2(v * (-1.0 / cutlass.math.log2e))
        tmp[i] = one / (one + cute.arch.exp(neg_one * v))
        tmp[i] *= tmp2[i]

    return tmp

class HopperFusedSwigluKernel:
    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ):
        """
        Initializes the configuration for a Hopper fused swiglu kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mnk = cluster_shape_mnk
        self.mma_inst_shape_mn = None
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if tile_shape_mnk[0] > 64 and tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_w1 = None
        self.num_mcast_ctas_w2 = None
        self.is_a_mcast = False
        self.is_w1_mcast = False
        self.is_w2_mcast = False

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = sm90_utils.SMEM_CAPACITY["sm90"]

        self.aw_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.w1_smem_layout_staged = None
        self.w2_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/W1/W2
        - Computing epilogue subtile
        - Setting up A/W1/W2/C stage counts in shared memory
        - Computing A/W1/W2/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")
        if self.tile_shape_mnk[2] not in [64]:
            raise ValueError("CTA tile shape K must be 64")

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_w1 = self.cluster_shape_mnk[0]
        self.num_mcast_ctas_w2 = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_w1_mcast = self.num_mcast_ctas_w1 > 1
        self.is_w2_mcast = self.num_mcast_ctas_w2 > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.aw_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.w_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.w1_smem_layout_staged,
            self.w2_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.w_dtype,
            self.w_layout,
            self.aw_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        w1: cute.Tensor,
        w2: cute.Tensor,
        c: cute.Tensor,
    ):
        """Execute the swiglu operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param w1: Input tensor W1
        :type w1: cute.Tensor
        :param w2: Input tensor W2
        :type w2: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.w_dtype = w1.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.w_layout = utils.LayoutEnum.from_tensor(w1)
        self.c_layout = utils.LayoutEnum.from_tensor(c)


        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.w_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.w_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.w_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.w_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError(f"a_dtype should be float16 or float8")

        self._setup_attributes()

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.w_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.w_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        tiled_mma_2 = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.w_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.w_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )

        tma_atom_w1, tma_tensor_w1 = self._make_tma_atoms_and_tensors(
            w1,
            self.w1_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_w2, tma_tensor_w2 = self._make_tma_atoms_and_tensors(
            w2,
            self.w2_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        grid = self._compute_grid(c, self.tile_shape_mnk, self.cluster_shape_mnk)
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.aw_stage * 2
            ]
            sa: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sw1: cute.struct.Align[
                cute.struct.MemRange[
                    self.w_dtype, cute.cosize(self.w1_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sw2: cute.struct.Align[
                cute.struct.MemRange[
                    self.w_dtype, cute.cosize(self.w2_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w2,
            tma_tensor_w2,
            tma_atom_c,
            tma_tensor_c,
            tiled_mma,
            tiled_mma_2,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.w1_smem_layout_staged,
            self.w2_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_w1: cute.CopyAtom,
        mW1_nkl: cute.Tensor,
        tma_atom_w2: cute.CopyAtom,
        mW2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_mma_2: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        w1_smem_layout_staged: cute.ComposedLayout,
        w2_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_w1: TMA copy atom for W1 tensor
        :type tma_atom_w1: cute.CopyAtom
        :param mW1_nkl: Input tensor W1
        :type mW1_nkl: cute.Tensor
        :param tma_atom_w2: TMA copy atom for W2 tensor
        :type tma_atom_w2: cute.CopyAtom
        :param mW2_nkl: Input tensor W2
        :type mW2_nkl: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mnl: Output tensor C
        :type mC_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param w1_smem_layout_staged: Shared memory layout for W1
        :type w1_smem_layout_staged: cute.ComposedLayout
        :param w2_smem_layout_staged: Shared memory layout for W2
        :type w2_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w2)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        tile_coord_mnkl = (bidx, bidy, None, bidz)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        w1_smem_layout = cute.slice_(w1_smem_layout_staged, (None, None, 0))
        w2_smem_layout = cute.slice_(w2_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.w_dtype, w1_smem_layout) + cute.size_in_bytes(self.w_dtype, w2_smem_layout)
        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count.
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.aw_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/W1/W2
        # ///////////////////////////////////////////////////////////////////////////////
        sa = storage.sa.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sw1 = storage.sw1.get_tensor(
            w1_smem_layout_staged.outer, swizzle=w1_smem_layout_staged.inner
        )
        sw2 = storage.sw2.get_tensor(
            w2_smem_layout_staged.outer, swizzle=w2_smem_layout_staged.inner
        )
        sc_ptr = cute.recast_ptr(
            sa.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sc = cute.make_tensor(sc_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        gA_mkl = cute.local_tile(
            mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
        )
        # (bN, bK, RestK)
        gW1_nkl = cute.local_tile(
            mW1_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        gW2_nkl = cute.local_tile(
            mW2_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
        )
        # (bM, bN)
        gC_mnl = cute.local_tile(
            mC_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/W/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mnl)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/W1/W2
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        sa_for_tma_partition = cute.group_modes(sa, 0, 2)
        gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
        tAsA, tAgA_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            a_cta_layout,
            sa_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load W1 partition_S/D
        w1_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        sw1_for_tma_partition = cute.group_modes(sw1, 0, 2)
        gW1_for_tma_partition = cute.group_modes(gW1_nkl, 0, 2)
        tW1sW1, tW1gW1_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w1,
            0,
            w1_cta_layout,
            sw1_for_tma_partition,
            gW1_for_tma_partition,
        )
        # TMA load W2 partition_S/D
        w2_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        sw2_for_tma_partition = cute.group_modes(sw2, 0, 2)
        gW2_for_tma_partition = cute.group_modes(gW2_nkl, 0, 2)
        tW2sW2, tW2gW2_nkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_w2,
            0,
            w2_cta_layout,
            sw2_for_tma_partition,
            gW2_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sa)
        tCsW1 = thr_mma.partition_B(sw1)
        tCsW2 = thr_mma.partition_B(sw2)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrW1 = tiled_mma.make_fragment_B(tCsW1)
        tCrW2 = tiled_mma.make_fragment_B(tCsW2)

        acc_shape = tCgC.shape
        accumulators = cute.make_fragment(acc_shape, self.acc_dtype)
        accumulator2 = cute.make_fragment(acc_shape, self.acc_dtype)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mkl, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.aw_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.aw_stage
        )
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/W1/W2 buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/W1/W2 buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tW1gW1_k = tW1gW1_nkl[(None, mainloop_producer_state.count)]
                tW1sW1_pipe = tW1sW1[(None, mainloop_producer_state.index)]

                tW2gW2_k = tW2gW2_nkl[(None, mainloop_producer_state.count)]
                tW2sW2_pipe = tW2sW2[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/W1/W2
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_w1,
                    tW1gW1_k,
                    tW1sW1_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_w2,
                    tW2gW2_k,
                    tW2sW2_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.aw_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.aw_stage
        )

        peek_aw_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_aw_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in range(k_pipe_mmas):
            # Wait for A/W1 buffer to be ready
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_aw_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrW1_1phase = tCrW1[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrW1_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                tCrW2_1phase = tCrW2[k_block_coord]
                cute.gemm(
                    tiled_mma_2,
                    accumulator2,
                    tCrA_1phase,
                    tCrW2_1phase,
                    accumulator2,)
                tiled_mma_2.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_aw_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_aw_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_aw_full_status
            )
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrW1_1phase = tCrW1[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrW1_1phase,
                    accumulators,
                )
                tCrW2_1phase = tCrW2[k_block_coord]
                cute.gemm(
                    tiled_mma_2,
                    accumulator2,
                    tCrA_1phase,
                    tCrW2_1phase,
                    accumulator2,
                )

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_aw_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_aw_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/W1/W2 buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/W1/W2 buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tW1gW1_k = tW1gW1_nkl[(None, mainloop_producer_state.count)]
                tW1sW1_pipe = tW1sW1[(None, mainloop_producer_state.index)]

                tW2gW2_k = tW2gW2_nkl[(None, mainloop_producer_state.count)]
                tW2sW2_pipe = tW2sW2[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/W1/W2
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_w1,
                    tW1gW1_k,
                    tW1sW1_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                cute.copy(
                    tma_atom_w2,
                    tW2gW2_k,
                    tW2sW2_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        # For cluster that has a single thread block, it might have more than one warp groups.
        # Wait for all warp groups in the thread block to finish, because smem for tensor A in
        # the mainloop is reused in the epilogue.
        cute.arch.sync_threads()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sc)
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)
        tRS_rAcc_2 = tiled_copy_r2s.retile(accumulator2)


        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sc))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
        tRS_rD1 = cute.make_fragment_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(sc, 0, 2)
        tcgc_for_tma_partition = cute.zipped_divide(gC_mnl, self.epi_tile)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tcgc_for_tma_partition,
        )

        epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
        epi_tile_shape = tcgc_for_tma_partition.shape[1]

        for epi_idx in cutlass.range(epi_tile_num, unroll=epi_tile_num):
            # Copy from accumulators to D registers
            for epi_v in range(size_tRS_rD):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                tRS_rD1[epi_v] = tRS_rAcc_2[epi_idx * size_tRS_rD + epi_v]

            # Type conversion
            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD.load()
            tRS_rD_out.store(acc_vec.to(self.c_dtype))

            tRS_rD1_out = cute.make_fragment_like(tRS_rD_layout, self.c_dtype)
            acc_vec = tRS_rD1.load()
            tRS_rD1_out.store(acc_vec.to(self.c_dtype))
            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            # Sigmoid + residual connection
            # TODO: This can be passed as a parameter to the kernel
            tRS_rD_out = sigmoid(tRS_rD_out.load(), tRS_rD1_out.load()) 
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            # barrier for sync
            cute.arch.barrier()

            # Get the global memory coordinate for the current epi tile.
            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)

            cute.arch.barrier()

        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        w_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/W1/W2/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param w1_dtype: Data type of operand W1.
        :type w_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/W operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab.
        epi_bytes = 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        w_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(w_shape) * w_dtype.width // 8
            + cute.size(w_shape) * w_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        w_dtype: type[cutlass.Numeric],
        w_layout: utils.LayoutEnum,
        aw_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param w_dtype: Data type for matrix W
        :type w_dtype: type[cutlass.Numeric]
        :param w_layout: Layout enum for matrix W
        :type w_layout: utils.LayoutEnum
        :param aw_stage: Number of stages for A/W tensors
        :type aw_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, W1, W2 and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        w_is_k_major = (
            w_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )

        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, aw_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        w1_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        w1_major_mode_size = tile_shape_mnk[2 if w_is_k_major else 1]
        w1_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                w_layout,
                w_dtype,
                w1_major_mode_size,
            ),
            w_dtype,
        )
        w1_smem_layout_staged = cute.tile_to_shape(
            w1_smem_layout_atom,
            cute.append(w1_smem_shape, aw_stage),
            order=(0, 1, 2) if w_is_k_major else (1, 0, 2),
        )

        w2_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        w2_major_mode_size = tile_shape_mnk[2 if w_is_k_major else 1]
        w2_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                w_layout,
                w_dtype,
                w2_major_mode_size,
            ),
            w_dtype,
        )
        w2_smem_layout_staged = cute.tile_to_shape(
            w2_smem_layout_atom,
            cute.append(w2_smem_shape, aw_stage),
            order=(0, 1, 2) if w_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, w1_smem_layout_staged, w2_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mnk: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mnk: Shape of each cluster in M, N, K dimensions.
        :type cluster_shape_mnk: tuple[int, int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mnk)
        grid = tuple(x * y for x, y in zip(clusters, cluster_shape_mnk))
        return grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), epi_tile
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or W)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        w_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        w_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param w_dtype: The data type of tensor W
        :type w_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param w_major: major mode of tensor W
        :type w_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested w_dtype
        if w_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype != cutlass.Float32:
            is_valid = False
        # tested c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == w_dtype for Float16
        if a_dtype.width == 16 and a_dtype != w_dtype:
            is_valid = False
        # make sure a_dtype.width == w_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != w_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            w_dtype.width == 8 and w_major != "k"
        ):
            is_valid = False

        return is_valid


def run_swiglu(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    w_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    w_major: str,
    c_major: str,
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
):
    """
    Prepare A/W/C tensors, launch GPU kernel, and reference checking.
    """

    print(f"Running Hopper Swiglu fused with:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {a_dtype}, W dtype: {w_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, W: {w_major}, C: {c_major}")
    print(f"Tile Shape: {tile_shape_mnk}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    # Skip unsupported types
    if not HopperFusedSwigluKernel.is_valid_dtypes(
        a_dtype, w_dtype, acc_dtype, c_dtype, a_major, w_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {w_dtype}, {acc_dtype}, {c_dtype}, {a_major=}, {w_major=}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(
        l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
    ):
        is_dynamic_layout = False
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=-4 if is_unsigned else -4, max_val=4 if is_unsigned else 4
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()
        f32_torch_tensor = torch_tensor.to(torch.float32)

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(
                leading_dim=(0 if is_mode0_major else 1)
            )
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    w1, mW1, w1_torch = create_and_permute_tensor(l, n, k, w_major == "n", w_dtype)
    w2, mW2, w2_torch = create_and_permute_tensor(l, n, k, w_major == "n", w_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)

    swiglu_kernel = HopperFusedSwigluKernel(acc_dtype, tile_shape_mnk, cluster_shape_mnk)

    # compile gemm kernel
    compiled_gemm = cute.compile(swiglu_kernel, mA, mW1, mW2, mC)
    # execution
    compiled_gemm(mA, mW1, mW2, mC)
    torch.cuda.synchronize()

    def torch_swiglu(x, w1, w2):
        """Sigmoid function for torch tensors."""
        mul1 = torch.einsum("mkl,nkl->mnl", x, w1)
        mul1 = 1 / (1 + torch.exp(-mul1))
        mul2 = torch.einsum("mkl,nkl->mnl", x, w2)
        return mul1 * mul2

    # Ref check
    ref = torch_swiglu(a, w1, w2).cpu()
    ref_c = ref.to(cutlass_torch.dtype(c_dtype))

    torch.testing.assert_close(c_torch.cpu(), ref_c, atol=tolerance, rtol=1e-03)

    def benchmark(callable, *, num_warmups, num_iterations, M, N, K):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        for _ in range(num_warmups):
            callable()

        start_event.record(stream=torch.cuda.current_stream())
        for _ in range(num_iterations):
            callable()
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        avg_time = elapsed_time / num_iterations
        print(f"Average execution time: {avg_time:.4f} ms")
        return avg_time

    def compute_tflops_ms(M: int, N: int, K: int, time_ms: float) -> float:
        """
        Compute achieved TFLOPs for a matrix multiplication kernel.
        Assumes execution time is in milliseconds.

        Args:
            M (int): Rows of matrix A.
            N (int): Columns of matrix B.
            K (int): Shared dimension.
            time_ms (float): Execution time in milliseconds.

        Returns:
            float: Achieved TFLOPs.
        """
        flops = (4 * M * N * K + 2 * M * N)  # 2 matmuls + sigmoid and add
        tflops = flops / (time_ms * 1e9)  # convert to TFLOPs
        return tflops

    from functools import partial
    avg_time = benchmark(
        partial(compiled_gemm, mA, mW1, mW2, mC),
        num_warmups=5,
        num_iterations=10,
        M=m,
        N=n,
        K=k,
    )
    tflops = compute_tflops_ms(m, n, k, avg_time)
    print(f"Achieved TFLOPs: {tflops:.2f}")

if __name__ == "__main__":
    args = parse_arguments()
    run_swiglu(
        args.mnkl,
        args.a_dtype,
        args.w_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.a_major,
        args.w_major,
        args.c_major,
        args.tile_shape_mnk,
        args.cluster_shape_mn,
        args.tolerance,
    )
    print("PASS")