#date: 2025-10-27T17:06:13Z
#url: https://api.github.com/gists/335f1f2ab52f79d2ffe9da2a13694f10
#owner: https://api.github.com/users/mattteochen

# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/tmpzs1g0hn_/6p/c6paalaw6ymqqxwpy3yqkd2sratjy75kwuwakoizzdscamrznvdq.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_4, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   hidden_states => convert_element_type_3
#   hidden_states_1 => mul_3
#   hidden_states_2 => mul_4
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_4 => convert_element_type_4
#   variance => mean
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[128256, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg7_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf0 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf0]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %convert_element_type_4), kwargs = {})
#   return %buf0,%mul_4
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 128256), "index out of bounds: 0 <= tmp5 < 128256")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.load(in_ptr0 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp15 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp15)
        tl.device_assert((0 <= tmp19) & (tmp19 < 128256), "index out of bounds: 0 <= tmp19 < 128256")
        tmp21 = tl.load(in_ptr1 + (r0_0 + 2048*tmp19), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = 2048.0
        tmp24 = (tmp11 / tmp23)
        tmp25 = 1e-05
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp13 * tmp29
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp30, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/6q/c6q3n2z4eyt5mlgkcbaswion4chjqn3m2unptu6d5w5dwenreeyo.py
# Topologically Sorted Source Nodes: [getitem_1, expand, , getitem_2, position_ids_expanded], Original ATen: [aten.unsqueeze, aten.expand, aten.bmm, aten._to_copy]
# Source node to ATen node mapping:
#    => mm_default, squeeze_dim, squeeze_dim_1
#   expand => expand
#   getitem_1 => unsqueeze, unsqueeze_1
#   getitem_2 => unsqueeze_2
#   position_ids_expanded => convert_element_type
# Graph fragment:
#   %arg3_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %expand_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=expand_2]
#   %unsqueeze : Tensor "f32[1, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg6_1, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %expand : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [1, -1, 1]), kwargs = {})
#   %squeeze_dim : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%expand_1, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_2, torch.float32), kwargs = {})
#   %squeeze_dim_1 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%expand_2, 0), kwargs = {})
#   %mm_default : Tensor "f32[32, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%squeeze_dim, %squeeze_dim_1), kwargs = {})
#   return %expand_2,%buf4
triton_poi_fused__to_copy_bmm_expand_unsqueeze_1 = async_compile.triton('triton_poi_fused__to_copy_bmm_expand_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bmm_expand_unsqueeze_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_bmm_expand_unsqueeze_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/t5/ct5s4hterk3bfezrdj6x4mjb2pf7e7qz4uut76iagevcoulshdgg.py
# Topologically Sorted Source Nodes: [, freqs, emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, linear_1, view_1, key_states, mul_7, x2_1, neg_1, x1_1, cat_2, mul_8, k_embed, index_copy_], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.index_copy]
# Source node to ATen node mapping:
#    => unsqueeze_default
#   cat_2 => cat_1
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   freqs => permute
#   index_copy_ => index_put
#   k_embed => add_2
#   key_states => permute_4
#   linear_1 => view_8
#   mul_7 => mul_7
#   mul_8 => mul_8
#   neg_1 => neg_1
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_5
#   view_1 => view_9
#   x1_1 => slice_3
#   x2_1 => slice_4
# Graph fragment:
#   %arg2_1 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_1 : Tensor "bf16[1, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_default : Tensor "f32[32, 1][1, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %index_put : Tensor "bf16[1, 8, 107, 64][54784, 6848, 64, 1]cuda:0" = PlaceHolder[target=index_put]
#   %unsqueeze_default : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %view_8 : Tensor "bf16[1, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 1, 512]), kwargs = {})
#   %view_9 : Tensor "bf16[1, 1, 8, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 1, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 8, 1, 64][512, 64, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [0, 2, 1, 3]), kwargs = {})
#   %mul_7 : Tensor "bf16[1, 8, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_4), kwargs = {})
#   %slice_4 : Tensor "bf16[1, 8, 1, 32][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 32, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "bf16[1, 8, 1, 32][256, 32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_4,), kwargs = {})
#   %slice_3 : Tensor "bf16[1, 8, 1, 32][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 32), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_3], -1), kwargs = {})
#   %mul_8 : Tensor "bf16[1, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_5), kwargs = {})
#   %add_2 : Tensor "bf16[1, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %index_put : Tensor "bf16[1, 8, 107, 64][54784, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg4_1, [None, None, %arg2_1], %add_2), kwargs = {})
#   return %buf7
triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + ((x2 % 32)), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([XBLOCK], 107, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 107), "index out of bounds: 0 <= tmp5 < 107")
    tmp9 = tl_math.cos(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp7 * tmp12
    tmp14 = x0
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tl.full([1], 32, tl.int64)
    tmp18 = tmp14 < tmp17
    tmp19 = tl.load(in_ptr1 + (32 + 64*x1 + (x0)), tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = -tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tmp14 >= tmp17
    tmp24 = tl.full([1], 64, tl.int64)
    tmp25 = tmp14 < tmp24
    tmp26 = tl.load(in_ptr1 + (64*x1 + ((-32) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp18, tmp22, tmp26)
    tmp28 = tl_math.sin(tmp8)
    tmp29 = tmp28 * tmp10
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp27 * tmp30
    tmp32 = tmp13 + tmp31
    tl.store(out_ptr0 + (x0 + 64*tmp5 + 6848*x1), tmp32, xmask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/vi/cviu57wssmctijqmr2mlctearcyoz7icdgneb7b6n5e2dv7iqmuq.py
# Topologically Sorted Source Nodes: [linear_2, view_2, value_states, index_copy__1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.index_copy]
# Source node to ATen node mapping:
#   index_copy__1 => index_put_1
#   linear_2 => view_11
#   value_states => permute_6
#   view_2 => view_12
# Graph fragment:
#   %arg2_1 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %mm_2 : Tensor "bf16[1, 512][512, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %index_put_1 : Tensor "bf16[1, 8, 107, 64][54784, 6848, 64, 1]cuda:0" = PlaceHolder[target=index_put_1]
#   %view_11 : Tensor "bf16[1, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 1, 512]), kwargs = {})
#   %view_12 : Tensor "bf16[1, 1, 8, 64][512, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, 1, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "bf16[1, 8, 1, 64][512, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %index_put_1 : Tensor "bf16[1, 8, 107, 64][54784, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%arg11_1, [None, None, %arg2_1], %permute_6), kwargs = {})
#   return %buf9
triton_poi_fused__unsafe_view_index_copy_transpose_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_index_copy_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_index_copy_transpose_view_3', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_index_copy_transpose_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tl.full([XBLOCK], 107, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 107), "index out of bounds: 0 <= tmp5 < 107")
    tl.store(out_ptr0 + (x0 + 64*tmp5 + 6848*x1), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/gd/cgdlz6cc7uvsuyhpw2hykkhdjlxnf46r2vltqwksqucfzfobxb5f.py
# Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, eq, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.eq, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
# Source node to ATen node mapping:
#    => unsqueeze_default
#   all_1 => any_2, logical_not, logical_not_1
#   attn_output => _scaled_dot_product_cudnn_attention
#   cat_1 => cat
#   causal_mask => mul
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   eq => eq
#   freqs => permute
#   invert => bitwise_not
#   key => clone_2, expand_5, unsqueeze_7, view_13
#   linear => view_5
#   mul_5 => mul_5
#   mul_6 => mul_6
#   neg => neg
#   q_embed => add_1
#   query_states => permute_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_5
#   value => clone_3, expand_7, unsqueeze_9, view_14
#   view => view_6
#   x1 => slice_1
#   x2 => slice_2
# Graph fragment:
#   %arg5_1 : Tensor "bf16[1, 1, 1, 107][107, 107, 107, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %any_2 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = PlaceHolder[target=any_2]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 32, 64][2048, 2048, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_1 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 32, 107, 64]), kwargs = {})
#   %unsqueeze_9 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_1, 2), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_9, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 32, 107, 64]), kwargs = {})
#   %eq : Tensor "b8[1, 1, 1, 107][107, 107, 107, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%arg5_1, -3.3895313892515355e+38), kwargs = {})
#   %logical_not : Tensor "b8[1, 1, 1, 107][107, 107, 107, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_2 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %bitwise_not : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_not_1,), kwargs = {})
#   %mul : Tensor "bf16[1, 1, 1, 107][107, 107, 107, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, %bitwise_not), kwargs = {})
#   %_scaled_dot_product_cudnn_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_cudnn_attention.default](args = (%add_1, %view_13, %view_14, %mul, False), kwargs = {scale: 0.125})
#   return %any_2,%buf14
triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4 = async_compile.triton('triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'tiling_scores': {'r0_': 642}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 107
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = -3.3895313892515355e+38
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2 == 0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = (tmp4 != 0)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, False)
    tmp9 = triton_helpers.any(tmp8, 1)[:, None].to(tl.int1)
    tmp10 = tmp9 == 0
    tmp11 = tmp10 == 0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp0 * tmp12
    tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp13, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/em/cemmp7onyo5ruusdwsgltp7rqs4a4m2nl26wglhtconaid3ankaq.py
# Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
# Source node to ATen node mapping:
#    => unsqueeze_default
#   all_1 => logical_not_1
#   attn_output => _scaled_dot_product_cudnn_attention
#   cat_1 => cat
#   causal_mask => mul
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   freqs => permute
#   invert => bitwise_not
#   key => clone_2, expand_5, unsqueeze_7, view_13
#   linear => view_5
#   mul_5 => mul_5
#   mul_6 => mul_6
#   neg => neg
#   q_embed => add_1
#   query_states => permute_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_5
#   value => clone_3, expand_7, unsqueeze_9, view_14
#   view => view_6
#   x1 => slice_1
#   x2 => slice_2
# Graph fragment:
#   %mm : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %mm_default : Tensor "f32[32, 1][1, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 32, 64][2048, 2048, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_1 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 32, 107, 64]), kwargs = {})
#   %unsqueeze_9 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_1, 2), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_9, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 32, 107, 64]), kwargs = {})
#   %logical_not_1 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %bitwise_not : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_not_1,), kwargs = {})
#   %mul : Tensor "bf16[1, 1, 1, 107][107, 107, 107, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, %bitwise_not), kwargs = {})
#   %_scaled_dot_product_cudnn_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_cudnn_attention.default](args = (%add_1, %view_13, %view_14, %mul, False), kwargs = {scale: 0.125})
#   return %buf11
triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'tiling_scores': {'x': 20736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((x2 % 32)), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = x0
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1], 32, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tl.load(in_ptr0 + (32 + 64*x1 + (x0)), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tmp7 >= tmp10
    tmp17 = tl.full([1], 64, tl.int64)
    tmp18 = tmp7 < tmp17
    tmp19 = tl.load(in_ptr0 + (64*x1 + ((-32) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp11, tmp15, tmp19)
    tmp21 = tl_math.sin(tmp1)
    tmp22 = tmp21 * tmp3
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp6 + tmp24
    tl.store(out_ptr0 + (x2), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/az/cazrc23plxynn5vbaimxnobiddwsvlp2ueys4r4zr6tyjv4nzcid.py
# Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
# Source node to ATen node mapping:
#    => unsqueeze_default
#   all_1 => logical_not_1
#   attn_output => _scaled_dot_product_cudnn_attention
#   cat_1 => cat
#   causal_mask => mul
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   freqs => permute
#   invert => bitwise_not
#   key => clone_2, expand_5, unsqueeze_7, view_13
#   linear => view_5
#   mul_5 => mul_5
#   mul_6 => mul_6
#   neg => neg
#   q_embed => add_1
#   query_states => permute_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_5
#   value => clone_3, expand_7, unsqueeze_9, view_14
#   view => view_6
#   x1 => slice_1
#   x2 => slice_2
# Graph fragment:
#   %buf7 : Tensor "bf16[1, 8, 107, 64][54784, 6848, 64, 1]cuda:0" = PlaceHolder[target=buf7]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 32, 64][2048, 2048, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 32, 1][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1, 32][32, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[1, 1, 1, 32][32, 1, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : Tensor "f32[1, 1, 2, 32][32, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 32][64, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 32, 1, 64][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 32, 1, 32][1024, 32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 32, 1, 32][2048, 64, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 32), kwargs = {})
#   %cat : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 64][64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1, 64][64, 64, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_1 : Tensor "bf16[1, 32, 1, 64][2048, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 32, 107, 64]), kwargs = {})
#   %unsqueeze_9 : Tensor "bf16[1, 8, 1, 107, 64][54784, 6848, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%index_put_1, 2), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 8, 4, 107, 64][54784, 6848, 0, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_9, [1, 8, 4, 107, 64]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 4, 107, 64][219136, 27392, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 32, 107, 64][219136, 6848, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 32, 107, 64]), kwargs = {})
#   %logical_not_1 : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %bitwise_not : Tensor "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_not_1,), kwargs = {})
#   %mul : Tensor "bf16[1, 1, 1, 107][107, 107, 107, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, %bitwise_not), kwargs = {})
#   %_scaled_dot_product_cudnn_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_cudnn_attention.default](args = (%add_1, %view_13, %view_14, %mul, False), kwargs = {scale: 0.125})
#   return %buf12
triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'tiling_scores': {'x': 876544}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 219136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6848)
    x1 = xindex // 6848
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6848*(x1 // 4)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/lt/cltvg2nwuwwrun62mxgrajqd7smcqk5iz43lfscyoeriueqbmbxp.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_4, rsqrt_1, hidden_states_7, to_8, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_4
#   attn_output_3 => view_17
#   hidden_states_5 => add_3
#   hidden_states_6 => convert_element_type_13
#   hidden_states_7 => mul_9
#   hidden_states_8 => mul_10
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_8 => convert_element_type_14
#   variance_1 => mean_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[128256, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_3 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg13_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf21 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf21]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 2048]), kwargs = {})
#   %add_3 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_13, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_9 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %rsqrt_1), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
#   %mul_10 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg13_1, %convert_element_type_14), kwargs = {})
#   return %buf21,%mul_10
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 128256), "index out of bounds: 0 <= tmp5 < 128256")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp16 = tl.load(in_ptr0 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp15 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp17 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp17)
        tl.device_assert((0 <= tmp21) & (tmp21 < 128256), "index out of bounds: 0 <= tmp21 < 128256")
        tmp23 = tl.load(in_ptr1 + (r0_0 + 2048*tmp21), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 2048.0
        tmp28 = (tmp13 / tmp27)
        tmp29 = 1e-05
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp15 * tmp33
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/3a/c3aad6cmqmr2kmwwr32vhuuv2o275drkl6qaobskdafvijd3keeo.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_19
#   linear_5 => view_21
#   mul_11 => mul_12
#   silu => convert_element_type_17, convert_element_type_18, mul_11, sigmoid
# Graph fragment:
#   %mm_4 : Tensor "bf16[1, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mm_5 : Tensor "bf16[1, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %view_19 : Tensor "bf16[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 1, 8192]), kwargs = {})
#   %convert_element_type_17 : Tensor "f32[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_17,), kwargs = {})
#   %mul_11 : Tensor "f32[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %sigmoid), kwargs = {})
#   %convert_element_type_18 : Tensor "bf16[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %view_21 : Tensor "bf16[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, 1, 8192]), kwargs = {})
#   %mul_12 : Tensor "bf16[1, 1, 8192][8192, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %view_21), kwargs = {})
#   return %mul_12
triton_poi_fused__unsafe_view_mul_silu_8 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'tiling_scores': {'x': 65536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/tmpzs1g0hn_/mz/cmzcja7qkregj4cht4cbafukisd5h5wqqrvoqonpqmzvfcsv4qrw.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_6, rsqrt_2, hidden_states_11, to_10, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_6 => add_6
#   attn_output_3 => view_17
#   down_proj => view_23
#   hidden_states_10 => convert_element_type_23
#   hidden_states_11 => mul_13
#   hidden_states_12 => mul_14
#   hidden_states_5 => add_3
#   hidden_states_9 => add_5
#   inputs_embeds => embedding
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_10 => convert_element_type_24
#   variance_2 => mean_2
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[128256, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_3 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %arg17_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %buf27 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf27]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 2048]), kwargs = {})
#   %add_3 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 1, 2048]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_23), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_23, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_6 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %rsqrt_2), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg17_1, %convert_element_type_24), kwargs = {})
#   return %buf27,%mul_14
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'A1B72E2CFE045B86677BD2BC442D5445F04C4F0FD097CD37A53DD63F1BDCFFB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 128256), "index out of bounds: 0 <= tmp5 < 128256")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp18 = tl.load(in_ptr0 + (0))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp17 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([XBLOCK, R0_BLOCK], 128256, tl.int32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp19 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp19)
        tl.device_assert((0 <= tmp23) & (tmp23 < 128256), "index out of bounds: 0 <= tmp23 < 128256")
        tmp25 = tl.load(in_ptr1 + (r0_0 + 2048*tmp23), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tmp25 + tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = 2048.0
        tmp32 = (tmp15 / tmp31)
        tmp33 = 1e-05
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp17 * tmp37
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp38, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1), (1, 1))
        assert_size_stride(arg1_1, (128256, 2048), (2048, 1))
        assert_size_stride(arg2_1, (1, ), (1, ))
        assert_size_stride(arg3_1, (1, 1), (1, 1))
        assert_size_stride(arg4_1, (1, 8, 107, 64), (54784, 6848, 64, 1))
        assert_size_stride(arg5_1, (1, 1, 1, 107), (107, 107, 107, 1))
        assert_size_stride(arg6_1, (32, ), (1, ))
        assert_size_stride(arg7_1, (2048, ), (1, ))
        assert_size_stride(arg8_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg9_1, (512, 2048), (2048, 1))
        assert_size_stride(arg10_1, (512, 2048), (2048, 1))
        assert_size_stride(arg11_1, (1, 8, 107, 64), (54784, 6848, 64, 1))
        assert_size_stride(arg12_1, (2048, 2048), (2048, 1))
        assert_size_stride(arg13_1, (2048, ), (1, ))
        assert_size_stride(arg14_1, (8192, 2048), (2048, 1))
        assert_size_stride(arg15_1, (8192, 2048), (2048, 1))
        assert_size_stride(arg16_1, (2048, 8192), (8192, 1))
        assert_size_stride(arg17_1, (2048, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_4, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0:1
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg7_1, buf1, 1, 2048, stream=stream0)
            del arg7_1
            buf2 = empty_strided_cuda((1, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_4, hidden_states_2, linear], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (0, 1), 0), reinterpret_tensor(arg8_1, (2048, 2048), (1, 2048), 0), out=buf2)
            del arg8_1
            buf3 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
            buf4 = reinterpret_tensor(buf3, (1, 1), (1, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [getitem_1, expand, , getitem_2, position_ids_expanded], Original ATen: [aten.unsqueeze, aten.expand, aten.bmm, aten._to_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy_bmm_expand_unsqueeze_1:3
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_bmm_expand_unsqueeze_1.run(buf4, arg3_1, 1, stream=stream0)
            del arg3_1
            buf5 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [getitem_1, expand, , getitem_2, position_ids_expanded], Original ATen: [aten.unsqueeze, aten.expand, aten.bmm, aten._to_copy]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(arg6_1, (32, 1), (1, 1), 0), buf4, out=buf5)
            del arg6_1
            del buf4
            buf6 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 512), (1, 2048), 0), out=buf6)
            del arg9_1
            # Topologically Sorted Source Nodes: [, freqs, emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, linear_1, view_1, key_states, mul_7, x2_1, neg_1, x1_1, cat_2, mul_8, k_embed, index_copy_], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.index_copy]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_cos_index_copy_mul_neg_sin_slice_transpose_unsqueeze_view_2.run(arg2_1, buf6, buf5, arg4_1, 512, stream=stream0)
            buf8 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg10_1, (2048, 512), (1, 2048), 0), out=buf8)
            del arg10_1
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, index_copy__1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.index_copy]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_index_copy_transpose_view_3:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_index_copy_transpose_view_3.run(arg2_1, buf8, arg11_1, 512, stream=stream0)
            del arg2_1
            del buf8
            buf14 = empty_strided_cuda((1, 1, 1, 107), (107, 107, 107, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, eq, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.eq, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4:9
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_eq_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(arg5_1, buf14, 1, 107, stream=stream0)
            del arg5_1
            buf11 = reinterpret_tensor(buf1, (1, 32, 1, 64), (2048, 64, 64, 1), 0); del buf1  # reuse
            # Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf2, buf5, buf11, 2048, stream=stream0)
            del buf2
            del buf5
            buf12 = empty_strided_cuda((1, 32, 107, 64), (219136, 6848, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6:11
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(arg4_1, buf12, 219136, stream=stream0)
            del arg4_1
            buf13 = empty_strided_cuda((1, 32, 107, 64), (219136, 6848, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6:12
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_cudnn_attention__to_copy__unsafe_view_add_all_bitwise_not_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(arg11_1, buf13, 219136, stream=stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [linear, view, query_states, , freqs, emb, cos, cos_1, cos_2, cos_3, mul_5, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_6, q_embed, key, value, all_1, invert, causal_mask, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.bmm, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.sin, aten.add, aten.expand, aten.clone, aten.all, aten.bitwise_not, aten._scaled_dot_product_cudnn_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_cudnn_attention.default:13
            buf15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(buf11, buf12, buf13, buf14, False, scale=0.125)
            del buf12
            del buf13
            del buf14
            buf16 = buf15[0]
            assert_size_stride(buf16, (1, 32, 1, 64), (2048, 64, 64, 1), 'torch.ops.aten._scaled_dot_product_cudnn_attention.default')
            assert_alignment(buf16, 16, 'torch.ops.aten._scaled_dot_product_cudnn_attention.default')
            del buf15
            buf20 = reinterpret_tensor(buf11, (1, 2048), (2048, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:14
            extern_kernels.mm(reinterpret_tensor(buf16, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 2048), (1, 2048), 0), out=buf20)
            del arg12_1
            buf22 = reinterpret_tensor(buf16, (1, 1, 2048), (2048, 2048, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_4, rsqrt_1, hidden_states_7, to_8, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7:15
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7.run(arg0_1, arg1_1, buf20, arg13_1, buf22, 1, 2048, stream=stream0)
            del arg13_1
            buf23 = empty_strided_cuda((1, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf22, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg14_1, (2048, 8192), (1, 2048), 0), out=buf23)
            del arg14_1
            buf24 = empty_strided_cuda((1, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:17
            extern_kernels.mm(reinterpret_tensor(buf22, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg15_1, (2048, 8192), (1, 2048), 0), out=buf24)
            del arg15_1
            buf25 = reinterpret_tensor(buf23, (1, 1, 8192), (8192, 8192, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:18
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf25, buf24, 8192, stream=stream0)
            del buf24
            buf26 = reinterpret_tensor(buf22, (1, 2048), (2048, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf25, (1, 8192), (0, 1), 0), reinterpret_tensor(arg16_1, (8192, 2048), (1, 8192), 0), out=buf26)
            del arg16_1
            del buf25
            buf28 = reinterpret_tensor(buf20, (1, 1, 2048), (2048, 2048, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_6, rsqrt_2, hidden_states_11, to_10, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9:20
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9.run(buf28, arg0_1, arg1_1, buf26, arg17_1, 1, 2048, stream=stream0)
            del arg0_1
            del arg17_1
            del buf26
            buf29 = empty_strided_cuda((1, 128256), (128256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_6, rsqrt_2, hidden_states_11, to_10, hidden_states_12, getitem_10, logits], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.slice, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf28, (1, 2048), (0, 1), 0), reinterpret_tensor(arg1_1, (2048, 128256), (1, 2048), 0), out=buf29)
            del arg1_1
            del buf28
        return (reinterpret_tensor(buf29, (1, 1, 128256), (128256, 128256, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((128256, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((1, 8, 107, 64), (54784, 6848, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1, 1, 1, 107), (107, 107, 107, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1, 8, 107, 64), (54784, 6848, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
