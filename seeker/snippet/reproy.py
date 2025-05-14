#date: 2025-05-14T16:53:31Z
#url: https://api.github.com/gists/f5c9afeb24247838e7fb7812b1f47bc7
#owner: https://api.github.com/users/masnesral

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch._decomp', 'torch._refs', 'torch.testing', 'torch._prims', 'torch.distributions'}
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_after', 'skipfiles_inline_module_allowlist', 'constant_functions', 'repro_level'}
torch._dynamo.config.reorderable_logging_functions = set()
# torch._dynamo.config.ignore_logger_methods = {<bound method Logger.warning of <Logger mlblox.latentlm.models.external_models.internvl2_5_2b.modeling_internlm2 (WARNING)>>}
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config.fallback_random = True
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = "**********"

torch._inductor.config.fallback_random = True

isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Mon_Apr__3_17:16:06_PDT_2023
# Cuda compilation tools, release 12.1, V12.1.105
# Build cuda_12.1.r12.1/compiler.32688072_0

# GPU Hardware Info:
# NVIDIA H200 : 8


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.true_graph_0 = <lambda>()
        #self.false_graph_0 = <lambda>()



    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184):
        slice_1 = torch.ops.aten.slice.Tensor(primals_1, 2, 0, 3)
        slice_2 = torch.ops.aten.slice.Tensor(primals_1, 2, 3, 9223372036854775807);  primals_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_1, -1)
        mul = torch.ops.aten.mul.Tensor(unsqueeze, primals_2);  primals_2 = None
        view = torch.ops.aten.view.default(mul, [16, 8192, -1]);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(unsqueeze, 0.5);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, 3.141592653589793);  mul_1 = None
        add = torch.ops.aten.add.Tensor(mul_2, primals_3);  mul_2 = primals_3 = None
        view_1 = torch.ops.aten.view.default(add, [16, 8192, -1]);  add = None
        cos = torch.ops.aten.cos.default(view)
        cos_1 = torch.ops.aten.cos.default(view_1)
        add_1 = torch.ops.aten.add.Tensor(cos, cos_1);  cos = cos_1 = None
        sin = torch.ops.aten.sin.default(view);  view = None
        sin_1 = torch.ops.aten.sin.default(view_1);  view_1 = None
        add_2 = torch.ops.aten.add.Tensor(sin, sin_1);  sin = sin_1 = None
        cat = torch.ops.aten.cat.default([slice_1, add_1, add_2], -1);  slice_1 = add_1 = add_2 = None
        cat_1 = torch.ops.aten.cat.default([cat, slice_2], -1);  cat = slice_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        expand = torch.ops.aten.expand.default(unsqueeze_1, [16, -1, -1]);  unsqueeze_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None
        view_2 = torch.ops.aten.view.default(convert_element_type_2, [131072, 774]);  convert_element_type_2 = None
        permute = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_2, [0, 2, 0, 0]);  view_2 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(permute, [0, 0, 0, 2]);  permute = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type, constant_pad_nd_default, constant_pad_nd_default_1);  convert_element_type = constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_3 = torch.ops.aten.view.default(addmm_default, [16, 8192, 768]);  addmm_default = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_6)
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = sigmoid = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        view_4 = torch.ops.aten.view.default(convert_element_type_7, [131072, 768]);  convert_element_type_7 = None
        permute_1 = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_1 = torch.ops.aten.addmm.default(convert_element_type_8, view_4, permute_1);  convert_element_type_8 = view_4 = permute_1 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [16, 8192, 768]);  addmm_1 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format)
        var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_3 = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_13, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_13, getitem_3);  convert_element_type_13 = getitem_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_6 = torch.ops.aten.view.default(convert_element_type_17, [8208, 768]);  convert_element_type_17 = None
        permute_2 = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_2 = torch.ops.aten.addmm.default(convert_element_type_15, view_6, permute_2);  convert_element_type_15 = view_6 = permute_2 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [16, 513, 768]);  addmm_2 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        view_8 = torch.ops.aten.view.default(convert_element_type_14, [131072, 768]);  convert_element_type_14 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        addmm_3 = torch.ops.aten.addmm.default(convert_element_type_21, view_8, permute_3);  convert_element_type_21 = permute_3 = None
        view_9 = torch.ops.aten.view.default(addmm_3, [16, 8192, 768]);  addmm_3 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        permute_4 = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        addmm_4 = torch.ops.aten.addmm.default(convert_element_type_26, view_8, permute_4);  convert_element_type_26 = permute_4 = None
        view_11 = torch.ops.aten.view.default(addmm_4, [16, 8192, 768]);  addmm_4 = None
        view_12 = torch.ops.aten.view.default(view_7, [16, 513, 12, -1]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        view_13 = torch.ops.aten.view.default(view_9, [16, 8192, 12, -1]);  view_9 = None
        permute_6 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        view_14 = torch.ops.aten.view.default(view_11, [16, 8192, 12, -1]);  view_11 = None
        permute_7 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_5, permute_6, permute_7, scale = 0.125);  permute_5 = permute_6 = permute_7 = None
        getitem_4 = _scaled_dot_product_flash_attention[0];  _scaled_dot_product_flash_attention = None
        permute_8 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        view_15 = torch.ops.aten.view.default(permute_8, [16, 513, 768]);  permute_8 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        view_16 = torch.ops.aten.view.default(view_15, [8208, 768]);  view_15 = None
        permute_9 = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        addmm_5 = torch.ops.aten.addmm.default(convert_element_type_31, view_16, permute_9);  convert_element_type_31 = view_16 = permute_9 = None
        view_17 = torch.ops.aten.view.default(addmm_5, [16, 513, 768]);  addmm_5 = None
        add_5 = torch.ops.aten.add.Tensor(expand, view_17);  expand = view_17 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_2[0]
        getitem_14 = var_mean_2[1];  var_mean_2 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_13, 1e-06);  getitem_13 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_5, getitem_14);  getitem_14 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_17, torch.bfloat16);  primals_17 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_18 = torch.ops.aten.view.default(convert_element_type_38, [8208, 768]);  convert_element_type_38 = None
        permute_10 = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_6 = torch.ops.aten.addmm.default(convert_element_type_36, view_18, permute_10);  convert_element_type_36 = view_18 = permute_10 = None
        view_19 = torch.ops.aten.view.default(addmm_6, [16, 513, 3072]);  addmm_6 = None
        convert_element_type_42 = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476);  convert_element_type_42 = None
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_7);  mul_7 = add_7 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        view_20 = torch.ops.aten.view.default(convert_element_type_43, [8208, 3072]);  convert_element_type_43 = None
        permute_11 = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        addmm_7 = torch.ops.aten.addmm.default(convert_element_type_44, view_20, permute_11);  convert_element_type_44 = view_20 = permute_11 = None
        view_21 = torch.ops.aten.view.default(addmm_7, [16, 513, 768]);  addmm_7 = None
        add_8 = torch.ops.aten.add.Tensor(add_5, view_21);  add_5 = view_21 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_15 = var_mean_3[0]
        getitem_16 = var_mean_3[1];  var_mean_3 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_15, 1e-06);  getitem_15 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_16);  getitem_16 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_51, [8208, 768]);  convert_element_type_51 = None
        permute_12 = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        addmm_8 = torch.ops.aten.addmm.default(convert_element_type_49, view_22, permute_12);  convert_element_type_49 = permute_12 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [16, 513, 1536]);  addmm_8 = None
        split = torch.ops.aten.split.Tensor(view_23, 768, -1);  view_23 = None
        getitem_17 = split[0]
        getitem_18 = split[1];  split = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        permute_13 = torch.ops.aten.permute.default(convert_element_type_56, [1, 0]);  convert_element_type_56 = None
        addmm_9 = torch.ops.aten.addmm.default(convert_element_type_55, view_22, permute_13);  convert_element_type_55 = view_22 = permute_13 = None
        view_25 = torch.ops.aten.view.default(addmm_9, [16, 513, 768]);  addmm_9 = None
        view_26 = torch.ops.aten.view.default(getitem_17, [16, 513, 12, -1]);  getitem_17 = None
        permute_14 = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        view_27 = torch.ops.aten.view.default(getitem_18, [16, 513, 12, -1]);  getitem_18 = None
        permute_15 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        view_28 = torch.ops.aten.view.default(view_25, [16, 513, 12, -1]);  view_25 = None
        permute_16 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_61, convert_element_type_61)
        mean = torch.ops.aten.mean.dim(mul_11, [-1], True);  mul_11 = None
        add_10 = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_61, rsqrt_4);  convert_element_type_61 = rsqrt_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, primals_25);  mul_12 = primals_25 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_63, convert_element_type_63)
        mean_1 = torch.ops.aten.mean.dim(mul_14, [-1], True);  mul_14 = None
        add_11 = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_63, rsqrt_5);  convert_element_type_63 = rsqrt_5 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, primals_26);  mul_15 = primals_26 = None
        convert_element_type_64 = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_62, convert_element_type_64, permute_16, scale = 0.125);  convert_element_type_62 = convert_element_type_64 = permute_16 = None
        getitem_19 = _scaled_dot_product_flash_attention_1[0];  _scaled_dot_product_flash_attention_1 = None
        permute_17 = torch.ops.aten.permute.default(getitem_19, [0, 2, 1, 3]);  getitem_19 = None
        view_29 = torch.ops.aten.view.default(permute_17, [16, 513, 768]);  permute_17 = None
        convert_element_type_65 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_66 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        view_30 = torch.ops.aten.view.default(view_29, [8208, 768]);  view_29 = None
        permute_18 = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_10 = torch.ops.aten.addmm.default(convert_element_type_65, view_30, permute_18);  convert_element_type_65 = view_30 = permute_18 = None
        view_31 = torch.ops.aten.view.default(addmm_10, [16, 513, 768]);  addmm_10 = None
        add_12 = torch.ops.aten.add.Tensor(add_8, view_31);  add_8 = view_31 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_4[0]
        getitem_29 = var_mean_4[1];  var_mean_4 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_12, getitem_29);  getitem_29 = None
        mul_17 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_6);  sub_4 = rsqrt_6 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_72, [8208, 768]);  convert_element_type_72 = None
        permute_19 = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_11 = torch.ops.aten.addmm.default(convert_element_type_70, view_32, permute_19);  convert_element_type_70 = view_32 = permute_19 = None
        view_33 = torch.ops.aten.view.default(addmm_11, [16, 513, 3072]);  addmm_11 = None
        convert_element_type_76 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.7071067811865476);  convert_element_type_76 = None
        erf_1 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_14 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        convert_element_type_78 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_77, [8208, 3072]);  convert_element_type_77 = None
        permute_20 = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        addmm_12 = torch.ops.aten.addmm.default(convert_element_type_78, view_34, permute_20);  convert_element_type_78 = view_34 = permute_20 = None
        view_35 = torch.ops.aten.view.default(addmm_12, [16, 513, 768]);  addmm_12 = None
        add_15 = torch.ops.aten.add.Tensor(add_12, view_35);  add_12 = view_35 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_5[0]
        getitem_31 = var_mean_5[1];  var_mean_5 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_15, getitem_31);  getitem_31 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_7);  sub_5 = rsqrt_7 = None
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_87, [8208, 768]);  convert_element_type_87 = None
        permute_21 = torch.ops.aten.permute.default(convert_element_type_86, [1, 0]);  convert_element_type_86 = None
        addmm_13 = torch.ops.aten.addmm.default(convert_element_type_85, view_36, permute_21);  convert_element_type_85 = view_36 = permute_21 = None
        view_37 = torch.ops.aten.view.default(addmm_13, [16, 513, 768]);  addmm_13 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        permute_22 = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_14 = torch.ops.aten.addmm.default(convert_element_type_91, view_8, permute_22);  convert_element_type_91 = permute_22 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [16, 8192, 768]);  addmm_14 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        permute_23 = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        addmm_15 = torch.ops.aten.addmm.default(convert_element_type_96, view_8, permute_23);  convert_element_type_96 = permute_23 = None
        view_41 = torch.ops.aten.view.default(addmm_15, [16, 8192, 768]);  addmm_15 = None
        view_42 = torch.ops.aten.view.default(view_37, [16, 513, 12, -1]);  view_37 = None
        permute_24 = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        view_43 = torch.ops.aten.view.default(view_39, [16, 8192, 12, -1]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        view_44 = torch.ops.aten.view.default(view_41, [16, 8192, 12, -1]);  view_41 = None
        permute_26 = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_24, permute_25, permute_26, scale = 0.125);  permute_24 = permute_25 = permute_26 = None
        getitem_34 = _scaled_dot_product_flash_attention_2[0];  _scaled_dot_product_flash_attention_2 = None
        permute_27 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_45 = torch.ops.aten.view.default(permute_27, [16, 513, 768]);  permute_27 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        view_46 = torch.ops.aten.view.default(view_45, [8208, 768]);  view_45 = None
        permute_28 = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        addmm_16 = torch.ops.aten.addmm.default(convert_element_type_101, view_46, permute_28);  convert_element_type_101 = view_46 = permute_28 = None
        view_47 = torch.ops.aten.view.default(addmm_16, [16, 513, 768]);  addmm_16 = None
        add_18 = torch.ops.aten.add.Tensor(add_15, view_47);  add_15 = view_47 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_43 = var_mean_7[0]
        getitem_44 = var_mean_7[1];  var_mean_7 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_43, 1e-06);  getitem_43 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_18, getitem_44);  getitem_44 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_9);  sub_7 = rsqrt_9 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_107 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_108, [8208, 768]);  convert_element_type_108 = None
        permute_29 = torch.ops.aten.permute.default(convert_element_type_107, [1, 0]);  convert_element_type_107 = None
        addmm_17 = torch.ops.aten.addmm.default(convert_element_type_106, view_48, permute_29);  convert_element_type_106 = view_48 = permute_29 = None
        view_49 = torch.ops.aten.view.default(addmm_17, [16, 513, 3072]);  addmm_17 = None
        convert_element_type_112 = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.5)
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.7071067811865476);  convert_element_type_112 = None
        erf_2 = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_20 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_24, add_20);  mul_24 = add_20 = None
        convert_element_type_113 = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        convert_element_type_114 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        view_50 = torch.ops.aten.view.default(convert_element_type_113, [8208, 3072]);  convert_element_type_113 = None
        permute_30 = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        addmm_18 = torch.ops.aten.addmm.default(convert_element_type_114, view_50, permute_30);  convert_element_type_114 = view_50 = permute_30 = None
        view_51 = torch.ops.aten.view.default(addmm_18, [16, 513, 768]);  addmm_18 = None
        add_21 = torch.ops.aten.add.Tensor(add_18, view_51);  add_18 = view_51 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_45 = var_mean_8[0]
        getitem_46 = var_mean_8[1];  var_mean_8 = None
        add_22 = torch.ops.aten.add.Tensor(getitem_45, 1e-06);  getitem_45 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_21, getitem_46);  getitem_46 = None
        mul_27 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_10);  sub_8 = rsqrt_10 = None
        convert_element_type_119 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        view_52 = torch.ops.aten.view.default(convert_element_type_121, [8208, 768]);  convert_element_type_121 = None
        permute_31 = torch.ops.aten.permute.default(convert_element_type_120, [1, 0]);  convert_element_type_120 = None
        addmm_19 = torch.ops.aten.addmm.default(convert_element_type_119, view_52, permute_31);  convert_element_type_119 = permute_31 = None
        view_53 = torch.ops.aten.view.default(addmm_19, [16, 513, 1536]);  addmm_19 = None
        split_1 = torch.ops.aten.split.Tensor(view_53, 768, -1);  view_53 = None
        getitem_47 = split_1[0]
        getitem_48 = split_1[1];  split_1 = None
        convert_element_type_125 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_126 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_32 = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_20 = torch.ops.aten.addmm.default(convert_element_type_125, view_52, permute_32);  convert_element_type_125 = view_52 = permute_32 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [16, 513, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(getitem_47, [16, 513, 12, -1]);  getitem_47 = None
        permute_33 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        view_57 = torch.ops.aten.view.default(getitem_48, [16, 513, 12, -1]);  getitem_48 = None
        permute_34 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        view_58 = torch.ops.aten.view.default(view_55, [16, 513, 12, -1]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        convert_element_type_131 = torch.ops.prims.convert_element_type.default(permute_33, torch.float32);  permute_33 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_131, convert_element_type_131)
        mean_2 = torch.ops.aten.mean.dim(mul_28, [-1], True);  mul_28 = None
        add_23 = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_29 = torch.ops.aten.mul.Tensor(convert_element_type_131, rsqrt_11);  convert_element_type_131 = rsqrt_11 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, primals_49);  mul_29 = primals_49 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_133, convert_element_type_133)
        mean_3 = torch.ops.aten.mean.dim(mul_31, [-1], True);  mul_31 = None
        add_24 = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_133, rsqrt_12);  convert_element_type_133 = rsqrt_12 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, primals_50);  mul_32 = primals_50 = None
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_132, convert_element_type_134, permute_35, scale = 0.125);  convert_element_type_132 = convert_element_type_134 = permute_35 = None
        getitem_49 = _scaled_dot_product_flash_attention_3[0];  _scaled_dot_product_flash_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_59 = torch.ops.aten.view.default(permute_36, [16, 513, 768]);  permute_36 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_136 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        view_60 = torch.ops.aten.view.default(view_59, [8208, 768]);  view_59 = None
        permute_37 = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        addmm_21 = torch.ops.aten.addmm.default(convert_element_type_135, view_60, permute_37);  convert_element_type_135 = view_60 = permute_37 = None
        view_61 = torch.ops.aten.view.default(addmm_21, [16, 513, 768]);  addmm_21 = None
        add_25 = torch.ops.aten.add.Tensor(add_21, view_61);  add_21 = view_61 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_9[0]
        getitem_59 = var_mean_9[1];  var_mean_9 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_25, getitem_59);  getitem_59 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_13);  sub_9 = rsqrt_13 = None
        convert_element_type_140 = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_141 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        convert_element_type_142 = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        view_62 = torch.ops.aten.view.default(convert_element_type_142, [8208, 768]);  convert_element_type_142 = None
        permute_38 = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        addmm_22 = torch.ops.aten.addmm.default(convert_element_type_140, view_62, permute_38);  convert_element_type_140 = view_62 = permute_38 = None
        view_63 = torch.ops.aten.view.default(addmm_22, [16, 513, 3072]);  addmm_22 = None
        convert_element_type_146 = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_35 = torch.ops.aten.mul.Tensor(convert_element_type_146, 0.5)
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_146, 0.7071067811865476);  convert_element_type_146 = None
        erf_3 = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_27 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_35, add_27);  mul_35 = add_27 = None
        convert_element_type_147 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        convert_element_type_148 = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        view_64 = torch.ops.aten.view.default(convert_element_type_147, [8208, 3072]);  convert_element_type_147 = None
        permute_39 = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        addmm_23 = torch.ops.aten.addmm.default(convert_element_type_148, view_64, permute_39);  convert_element_type_148 = view_64 = permute_39 = None
        view_65 = torch.ops.aten.view.default(addmm_23, [16, 513, 768]);  addmm_23 = None
        add_28 = torch.ops.aten.add.Tensor(add_25, view_65);  add_25 = view_65 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_10[0]
        getitem_61 = var_mean_10[1];  var_mean_10 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_28, getitem_61);  getitem_61 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_14);  sub_10 = rsqrt_14 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_156 = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_157 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        view_66 = torch.ops.aten.view.default(convert_element_type_157, [8208, 768]);  convert_element_type_157 = None
        permute_40 = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        addmm_24 = torch.ops.aten.addmm.default(convert_element_type_155, view_66, permute_40);  convert_element_type_155 = view_66 = permute_40 = None
        view_67 = torch.ops.aten.view.default(addmm_24, [16, 513, 768]);  addmm_24 = None
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        permute_41 = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        addmm_25 = torch.ops.aten.addmm.default(convert_element_type_161, view_8, permute_41);  convert_element_type_161 = permute_41 = None
        view_69 = torch.ops.aten.view.default(addmm_25, [16, 8192, 768]);  addmm_25 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_42 = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        addmm_26 = torch.ops.aten.addmm.default(convert_element_type_166, view_8, permute_42);  convert_element_type_166 = permute_42 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [16, 8192, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_67, [16, 513, 12, -1]);  view_67 = None
        permute_43 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        view_73 = torch.ops.aten.view.default(view_69, [16, 8192, 12, -1]);  view_69 = None
        permute_44 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        view_74 = torch.ops.aten.view.default(view_71, [16, 8192, 12, -1]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_43, permute_44, permute_45, scale = 0.125);  permute_43 = permute_44 = permute_45 = None
        getitem_64 = _scaled_dot_product_flash_attention_4[0];  _scaled_dot_product_flash_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_64, [0, 2, 1, 3]);  getitem_64 = None
        view_75 = torch.ops.aten.view.default(permute_46, [16, 513, 768]);  permute_46 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        view_76 = torch.ops.aten.view.default(view_75, [8208, 768]);  view_75 = None
        permute_47 = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_27 = torch.ops.aten.addmm.default(convert_element_type_171, view_76, permute_47);  convert_element_type_171 = view_76 = permute_47 = None
        view_77 = torch.ops.aten.view.default(addmm_27, [16, 513, 768]);  addmm_27 = None
        add_31 = torch.ops.aten.add.Tensor(add_28, view_77);  add_28 = view_77 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_73 = var_mean_12[0]
        getitem_74 = var_mean_12[1];  var_mean_12 = None
        add_32 = torch.ops.aten.add.Tensor(getitem_73, 1e-06);  getitem_73 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_31, getitem_74);  getitem_74 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_16);  sub_12 = rsqrt_16 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        view_78 = torch.ops.aten.view.default(convert_element_type_178, [8208, 768]);  convert_element_type_178 = None
        permute_48 = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        addmm_28 = torch.ops.aten.addmm.default(convert_element_type_176, view_78, permute_48);  convert_element_type_176 = view_78 = permute_48 = None
        view_79 = torch.ops.aten.view.default(addmm_28, [16, 513, 3072]);  addmm_28 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        mul_41 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_42 = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_4 = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_33 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        view_80 = torch.ops.aten.view.default(convert_element_type_183, [8208, 3072]);  convert_element_type_183 = None
        permute_49 = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_29 = torch.ops.aten.addmm.default(convert_element_type_184, view_80, permute_49);  convert_element_type_184 = view_80 = permute_49 = None
        view_81 = torch.ops.aten.view.default(addmm_29, [16, 513, 768]);  addmm_29 = None
        add_34 = torch.ops.aten.add.Tensor(add_31, view_81);  add_31 = view_81 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_75 = var_mean_13[0]
        getitem_76 = var_mean_13[1];  var_mean_13 = None
        add_35 = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_34, getitem_76);  getitem_76 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_17);  sub_13 = rsqrt_17 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        convert_element_type_191 = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        view_82 = torch.ops.aten.view.default(convert_element_type_191, [8208, 768]);  convert_element_type_191 = None
        permute_50 = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        addmm_30 = torch.ops.aten.addmm.default(convert_element_type_189, view_82, permute_50);  convert_element_type_189 = permute_50 = None
        view_83 = torch.ops.aten.view.default(addmm_30, [16, 513, 1536]);  addmm_30 = None
        split_2 = torch.ops.aten.split.Tensor(view_83, 768, -1);  view_83 = None
        getitem_77 = split_2[0]
        getitem_78 = split_2[1];  split_2 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_196 = torch.ops.prims.convert_element_type.default(primals_71, torch.bfloat16);  primals_71 = None
        permute_51 = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        addmm_31 = torch.ops.aten.addmm.default(convert_element_type_195, view_82, permute_51);  convert_element_type_195 = view_82 = permute_51 = None
        view_85 = torch.ops.aten.view.default(addmm_31, [16, 513, 768]);  addmm_31 = None
        view_86 = torch.ops.aten.view.default(getitem_77, [16, 513, 12, -1]);  getitem_77 = None
        permute_52 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        view_87 = torch.ops.aten.view.default(getitem_78, [16, 513, 12, -1]);  getitem_78 = None
        permute_53 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        view_88 = torch.ops.aten.view.default(view_85, [16, 513, 12, -1]);  view_85 = None
        permute_54 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        convert_element_type_201 = torch.ops.prims.convert_element_type.default(permute_52, torch.float32);  permute_52 = None
        mul_45 = torch.ops.aten.mul.Tensor(convert_element_type_201, convert_element_type_201)
        mean_4 = torch.ops.aten.mean.dim(mul_45, [-1], True);  mul_45 = None
        add_36 = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_201, rsqrt_18);  convert_element_type_201 = rsqrt_18 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, primals_73);  mul_46 = primals_73 = None
        convert_element_type_202 = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        convert_element_type_203 = torch.ops.prims.convert_element_type.default(permute_53, torch.float32);  permute_53 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_203, convert_element_type_203)
        mean_5 = torch.ops.aten.mean.dim(mul_48, [-1], True);  mul_48 = None
        add_37 = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_203, rsqrt_19);  convert_element_type_203 = rsqrt_19 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_74);  mul_49 = primals_74 = None
        convert_element_type_204 = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None
        _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_202, convert_element_type_204, permute_54, scale = 0.125);  convert_element_type_202 = convert_element_type_204 = permute_54 = None
        getitem_79 = _scaled_dot_product_flash_attention_5[0];  _scaled_dot_product_flash_attention_5 = None
        permute_55 = torch.ops.aten.permute.default(getitem_79, [0, 2, 1, 3]);  getitem_79 = None
        view_89 = torch.ops.aten.view.default(permute_55, [16, 513, 768]);  permute_55 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        view_90 = torch.ops.aten.view.default(view_89, [8208, 768]);  view_89 = None
        permute_56 = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_32 = torch.ops.aten.addmm.default(convert_element_type_205, view_90, permute_56);  convert_element_type_205 = view_90 = permute_56 = None
        view_91 = torch.ops.aten.view.default(addmm_32, [16, 513, 768]);  addmm_32 = None
        add_38 = torch.ops.aten.add.Tensor(add_34, view_91);  add_34 = view_91 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_14[0]
        getitem_89 = var_mean_14[1];  var_mean_14 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_38, getitem_89);  getitem_89 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_20);  sub_14 = rsqrt_20 = None
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_212, [8208, 768]);  convert_element_type_212 = None
        permute_57 = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        addmm_33 = torch.ops.aten.addmm.default(convert_element_type_210, view_92, permute_57);  convert_element_type_210 = view_92 = permute_57 = None
        view_93 = torch.ops.aten.view.default(addmm_33, [16, 513, 3072]);  addmm_33 = None
        convert_element_type_216 = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_52 = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.5)
        mul_53 = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.7071067811865476);  convert_element_type_216 = None
        erf_5 = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_40 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_52, add_40);  mul_52 = add_40 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        convert_element_type_218 = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_219 = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        view_94 = torch.ops.aten.view.default(convert_element_type_217, [8208, 3072]);  convert_element_type_217 = None
        permute_58 = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_34 = torch.ops.aten.addmm.default(convert_element_type_218, view_94, permute_58);  convert_element_type_218 = view_94 = permute_58 = None
        view_95 = torch.ops.aten.view.default(addmm_34, [16, 513, 768]);  addmm_34 = None
        add_41 = torch.ops.aten.add.Tensor(add_38, view_95);  add_38 = view_95 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_90 = var_mean_15[0]
        getitem_91 = var_mean_15[1];  var_mean_15 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_41, getitem_91);  getitem_91 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_21);  sub_15 = rsqrt_21 = None
        convert_element_type_223 = torch.ops.prims.convert_element_type.default(primals_82, torch.bfloat16);  primals_82 = None
        convert_element_type_224 = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        view_96 = torch.ops.aten.view.default(convert_element_type_225, [8208, 768]);  convert_element_type_225 = None
        permute_59 = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        addmm_35 = torch.ops.aten.addmm.default(convert_element_type_223, view_96, permute_59);  convert_element_type_223 = permute_59 = None
        view_97 = torch.ops.aten.view.default(addmm_35, [16, 513, 1536]);  addmm_35 = None
        split_3 = torch.ops.aten.split.Tensor(view_97, 768, -1);  view_97 = None
        getitem_92 = split_3[0]
        getitem_93 = split_3[1];  split_3 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convert_element_type_230 = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        permute_60 = torch.ops.aten.permute.default(convert_element_type_230, [1, 0]);  convert_element_type_230 = None
        addmm_36 = torch.ops.aten.addmm.default(convert_element_type_229, view_96, permute_60);  convert_element_type_229 = view_96 = permute_60 = None
        view_99 = torch.ops.aten.view.default(addmm_36, [16, 513, 768]);  addmm_36 = None
        view_100 = torch.ops.aten.view.default(getitem_92, [16, 513, 12, -1]);  getitem_92 = None
        permute_61 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        view_101 = torch.ops.aten.view.default(getitem_93, [16, 513, 12, -1]);  getitem_93 = None
        permute_62 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        view_102 = torch.ops.aten.view.default(view_99, [16, 513, 12, -1]);  view_99 = None
        permute_63 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(permute_61, torch.float32);  permute_61 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_235, convert_element_type_235)
        mean_6 = torch.ops.aten.mean.dim(mul_56, [-1], True);  mul_56 = None
        add_43 = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_57 = torch.ops.aten.mul.Tensor(convert_element_type_235, rsqrt_22);  convert_element_type_235 = rsqrt_22 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, primals_85);  mul_57 = primals_85 = None
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(permute_62, torch.float32);  permute_62 = None
        mul_59 = torch.ops.aten.mul.Tensor(convert_element_type_237, convert_element_type_237)
        mean_7 = torch.ops.aten.mean.dim(mul_59, [-1], True);  mul_59 = None
        add_44 = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_60 = torch.ops.aten.mul.Tensor(convert_element_type_237, rsqrt_23);  convert_element_type_237 = rsqrt_23 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, primals_86);  mul_60 = primals_86 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_236, convert_element_type_238, permute_63, scale = 0.125);  convert_element_type_236 = convert_element_type_238 = permute_63 = None
        getitem_94 = _scaled_dot_product_flash_attention_6[0];  _scaled_dot_product_flash_attention_6 = None
        permute_64 = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None
        view_103 = torch.ops.aten.view.default(permute_64, [16, 513, 768]);  permute_64 = None
        convert_element_type_239 = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_240 = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        view_104 = torch.ops.aten.view.default(view_103, [8208, 768]);  view_103 = None
        permute_65 = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_37 = torch.ops.aten.addmm.default(convert_element_type_239, view_104, permute_65);  convert_element_type_239 = view_104 = permute_65 = None
        view_105 = torch.ops.aten.view.default(addmm_37, [16, 513, 768]);  addmm_37 = None
        add_45 = torch.ops.aten.add.Tensor(add_41, view_105);  add_41 = view_105 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_103 = var_mean_16[0]
        getitem_104 = var_mean_16[1];  var_mean_16 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_103, 1e-06);  getitem_103 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_45, getitem_104);  getitem_104 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_24);  sub_16 = rsqrt_24 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_245 = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_246 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_246, [8208, 768]);  convert_element_type_246 = None
        permute_66 = torch.ops.aten.permute.default(convert_element_type_245, [1, 0]);  convert_element_type_245 = None
        addmm_38 = torch.ops.aten.addmm.default(convert_element_type_244, view_106, permute_66);  convert_element_type_244 = view_106 = permute_66 = None
        view_107 = torch.ops.aten.view.default(addmm_38, [16, 513, 3072]);  addmm_38 = None
        convert_element_type_250 = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_250, 0.5)
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_250, 0.7071067811865476);  convert_element_type_250 = None
        erf_6 = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_47 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_63, add_47);  mul_63 = add_47 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        convert_element_type_252 = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_253 = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        view_108 = torch.ops.aten.view.default(convert_element_type_251, [8208, 3072]);  convert_element_type_251 = None
        permute_67 = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_39 = torch.ops.aten.addmm.default(convert_element_type_252, view_108, permute_67);  convert_element_type_252 = view_108 = permute_67 = None
        view_109 = torch.ops.aten.view.default(addmm_39, [16, 513, 768]);  addmm_39 = None
        add_48 = torch.ops.aten.add.Tensor(add_45, view_109);  add_45 = view_109 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_105 = var_mean_17[0]
        getitem_106 = var_mean_17[1];  var_mean_17 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_105, 1e-06);  getitem_105 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_48, getitem_106);  getitem_106 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_25);  sub_17 = rsqrt_25 = None
        convert_element_type_257 = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convert_element_type_258 = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        view_110 = torch.ops.aten.view.default(convert_element_type_259, [8208, 768]);  convert_element_type_259 = None
        permute_68 = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        addmm_40 = torch.ops.aten.addmm.default(convert_element_type_257, view_110, permute_68);  convert_element_type_257 = permute_68 = None
        view_111 = torch.ops.aten.view.default(addmm_40, [16, 513, 1536]);  addmm_40 = None
        split_4 = torch.ops.aten.split.Tensor(view_111, 768, -1);  view_111 = None
        getitem_107 = split_4[0]
        getitem_108 = split_4[1];  split_4 = None
        convert_element_type_263 = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        permute_69 = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_41 = torch.ops.aten.addmm.default(convert_element_type_263, view_110, permute_69);  convert_element_type_263 = view_110 = permute_69 = None
        view_113 = torch.ops.aten.view.default(addmm_41, [16, 513, 768]);  addmm_41 = None
        view_114 = torch.ops.aten.view.default(getitem_107, [16, 513, 12, -1]);  getitem_107 = None
        permute_70 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        view_115 = torch.ops.aten.view.default(getitem_108, [16, 513, 12, -1]);  getitem_108 = None
        permute_71 = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        view_116 = torch.ops.aten.view.default(view_113, [16, 513, 12, -1]);  view_113 = None
        permute_72 = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_269, convert_element_type_269)
        mean_8 = torch.ops.aten.mean.dim(mul_67, [-1], True);  mul_67 = None
        add_50 = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_269, rsqrt_26);  convert_element_type_269 = rsqrt_26 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, primals_97);  mul_68 = primals_97 = None
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(permute_71, torch.float32);  permute_71 = None
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_271, convert_element_type_271)
        mean_9 = torch.ops.aten.mean.dim(mul_70, [-1], True);  mul_70 = None
        add_51 = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_271, rsqrt_27);  convert_element_type_271 = rsqrt_27 = None
        mul_72 = torch.ops.aten.mul.Tensor(mul_71, primals_98);  mul_71 = primals_98 = None
        convert_element_type_272 = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_270, convert_element_type_272, permute_72, scale = 0.125);  convert_element_type_270 = convert_element_type_272 = permute_72 = None
        getitem_109 = _scaled_dot_product_flash_attention_7[0];  _scaled_dot_product_flash_attention_7 = None
        permute_73 = torch.ops.aten.permute.default(getitem_109, [0, 2, 1, 3]);  getitem_109 = None
        view_117 = torch.ops.aten.view.default(permute_73, [16, 513, 768]);  permute_73 = None
        convert_element_type_273 = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_274 = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        view_118 = torch.ops.aten.view.default(view_117, [8208, 768]);  view_117 = None
        permute_74 = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        addmm_42 = torch.ops.aten.addmm.default(convert_element_type_273, view_118, permute_74);  convert_element_type_273 = view_118 = permute_74 = None
        view_119 = torch.ops.aten.view.default(addmm_42, [16, 513, 768]);  addmm_42 = None
        add_52 = torch.ops.aten.add.Tensor(add_48, view_119);  add_48 = view_119 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_118 = var_mean_18[0]
        getitem_119 = var_mean_18[1];  var_mean_18 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_52, getitem_119);  getitem_119 = None
        mul_73 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_28);  sub_18 = rsqrt_28 = None
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        convert_element_type_280 = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        view_120 = torch.ops.aten.view.default(convert_element_type_280, [8208, 768]);  convert_element_type_280 = None
        permute_75 = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        addmm_43 = torch.ops.aten.addmm.default(convert_element_type_278, view_120, permute_75);  convert_element_type_278 = view_120 = permute_75 = None
        view_121 = torch.ops.aten.view.default(addmm_43, [16, 513, 3072]);  addmm_43 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        mul_74 = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.5)
        mul_75 = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476);  convert_element_type_284 = None
        erf_7 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_54 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_74, add_54);  mul_74 = add_54 = None
        convert_element_type_285 = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        view_122 = torch.ops.aten.view.default(convert_element_type_285, [8208, 3072]);  convert_element_type_285 = None
        permute_76 = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_44 = torch.ops.aten.addmm.default(convert_element_type_286, view_122, permute_76);  convert_element_type_286 = view_122 = permute_76 = None
        view_123 = torch.ops.aten.view.default(addmm_44, [16, 513, 768]);  addmm_44 = None
        add_55 = torch.ops.aten.add.Tensor(add_52, view_123);  add_52 = view_123 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_120 = var_mean_19[0]
        getitem_121 = var_mean_19[1];  var_mean_19 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_55, getitem_121);  getitem_121 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_29);  sub_19 = rsqrt_29 = None
        convert_element_type_293 = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_294 = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_295 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        view_124 = torch.ops.aten.view.default(convert_element_type_295, [8208, 768]);  convert_element_type_295 = None
        permute_77 = torch.ops.aten.permute.default(convert_element_type_294, [1, 0]);  convert_element_type_294 = None
        addmm_45 = torch.ops.aten.addmm.default(convert_element_type_293, view_124, permute_77);  convert_element_type_293 = view_124 = permute_77 = None
        view_125 = torch.ops.aten.view.default(addmm_45, [16, 513, 768]);  addmm_45 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_300 = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        permute_78 = torch.ops.aten.permute.default(convert_element_type_300, [1, 0]);  convert_element_type_300 = None
        addmm_46 = torch.ops.aten.addmm.default(convert_element_type_299, view_8, permute_78);  convert_element_type_299 = permute_78 = None
        view_127 = torch.ops.aten.view.default(addmm_46, [16, 8192, 768]);  addmm_46 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_305 = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_79 = torch.ops.aten.permute.default(convert_element_type_305, [1, 0]);  convert_element_type_305 = None
        addmm_47 = torch.ops.aten.addmm.default(convert_element_type_304, view_8, permute_79);  convert_element_type_304 = view_8 = permute_79 = None
        view_129 = torch.ops.aten.view.default(addmm_47, [16, 8192, 768]);  addmm_47 = None
        view_130 = torch.ops.aten.view.default(view_125, [16, 513, 12, -1]);  view_125 = None
        permute_80 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        view_131 = torch.ops.aten.view.default(view_127, [16, 8192, 12, -1]);  view_127 = None
        permute_81 = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        view_132 = torch.ops.aten.view.default(view_129, [16, 8192, 12, -1]);  view_129 = None
        permute_82 = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_80, permute_81, permute_82, scale = 0.125);  permute_80 = permute_81 = permute_82 = None
        getitem_124 = _scaled_dot_product_flash_attention_8[0];  _scaled_dot_product_flash_attention_8 = None
        permute_83 = torch.ops.aten.permute.default(getitem_124, [0, 2, 1, 3]);  getitem_124 = None
        view_133 = torch.ops.aten.view.default(permute_83, [16, 513, 768]);  permute_83 = None
        convert_element_type_309 = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_310 = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_134 = torch.ops.aten.view.default(view_133, [8208, 768]);  view_133 = None
        permute_84 = torch.ops.aten.permute.default(convert_element_type_310, [1, 0]);  convert_element_type_310 = None
        addmm_48 = torch.ops.aten.addmm.default(convert_element_type_309, view_134, permute_84);  convert_element_type_309 = view_134 = permute_84 = None
        view_135 = torch.ops.aten.view.default(addmm_48, [16, 513, 768]);  addmm_48 = None
        add_58 = torch.ops.aten.add.Tensor(add_55, view_135);  add_55 = view_135 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_133 = var_mean_21[0]
        getitem_134 = var_mean_21[1];  var_mean_21 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_133, 1e-06);  getitem_133 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_58, getitem_134);  getitem_134 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_31);  sub_21 = rsqrt_31 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        convert_element_type_315 = torch.ops.prims.convert_element_type.default(primals_113, torch.bfloat16);  primals_113 = None
        convert_element_type_316 = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_136 = torch.ops.aten.view.default(convert_element_type_316, [8208, 768]);  convert_element_type_316 = None
        permute_85 = torch.ops.aten.permute.default(convert_element_type_315, [1, 0]);  convert_element_type_315 = None
        addmm_49 = torch.ops.aten.addmm.default(convert_element_type_314, view_136, permute_85);  convert_element_type_314 = view_136 = permute_85 = None
        view_137 = torch.ops.aten.view.default(addmm_49, [16, 513, 3072]);  addmm_49 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_320, 0.5)
        mul_81 = torch.ops.aten.mul.Tensor(convert_element_type_320, 0.7071067811865476);  convert_element_type_320 = None
        erf_8 = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_60 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_82 = torch.ops.aten.mul.Tensor(mul_80, add_60);  mul_80 = add_60 = None
        convert_element_type_321 = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16);  mul_82 = None
        convert_element_type_322 = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_323 = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        view_138 = torch.ops.aten.view.default(convert_element_type_321, [8208, 3072]);  convert_element_type_321 = None
        permute_86 = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        addmm_50 = torch.ops.aten.addmm.default(convert_element_type_322, view_138, permute_86);  convert_element_type_322 = view_138 = permute_86 = None
        view_139 = torch.ops.aten.view.default(addmm_50, [16, 513, 768]);  addmm_50 = None
        add_61 = torch.ops.aten.add.Tensor(add_58, view_139);  add_58 = view_139 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_135 = var_mean_22[0]
        getitem_136 = var_mean_22[1];  var_mean_22 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_135, 1e-06);  getitem_135 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_61, getitem_136);  getitem_136 = None
        mul_83 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_32);  sub_22 = rsqrt_32 = None
        convert_element_type_327 = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_328 = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_329 = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        view_140 = torch.ops.aten.view.default(convert_element_type_329, [8208, 768]);  convert_element_type_329 = None
        permute_87 = torch.ops.aten.permute.default(convert_element_type_328, [1, 0]);  convert_element_type_328 = None
        addmm_51 = torch.ops.aten.addmm.default(convert_element_type_327, view_140, permute_87);  convert_element_type_327 = permute_87 = None
        view_141 = torch.ops.aten.view.default(addmm_51, [16, 513, 1536]);  addmm_51 = None
        split_5 = torch.ops.aten.split.Tensor(view_141, 768, -1);  view_141 = None
        getitem_137 = split_5[0]
        getitem_138 = split_5[1];  split_5 = None
        convert_element_type_333 = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_334 = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        permute_88 = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        addmm_52 = torch.ops.aten.addmm.default(convert_element_type_333, view_140, permute_88);  convert_element_type_333 = view_140 = permute_88 = None
        view_143 = torch.ops.aten.view.default(addmm_52, [16, 513, 768]);  addmm_52 = None
        view_144 = torch.ops.aten.view.default(getitem_137, [16, 513, 12, -1]);  getitem_137 = None
        permute_89 = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        view_145 = torch.ops.aten.view.default(getitem_138, [16, 513, 12, -1]);  getitem_138 = None
        permute_90 = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        view_146 = torch.ops.aten.view.default(view_143, [16, 513, 12, -1]);  view_143 = None
        permute_91 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        convert_element_type_339 = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_339, convert_element_type_339)
        mean_10 = torch.ops.aten.mean.dim(mul_84, [-1], True);  mul_84 = None
        add_63 = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_339, rsqrt_33);  convert_element_type_339 = rsqrt_33 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, primals_121);  mul_85 = primals_121 = None
        convert_element_type_340 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        convert_element_type_341 = torch.ops.prims.convert_element_type.default(permute_90, torch.float32);  permute_90 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_341, convert_element_type_341)
        mean_11 = torch.ops.aten.mean.dim(mul_87, [-1], True);  mul_87 = None
        add_64 = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_341, rsqrt_34);  convert_element_type_341 = rsqrt_34 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, primals_122);  mul_88 = primals_122 = None
        convert_element_type_342 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_340, convert_element_type_342, permute_91, scale = 0.125);  convert_element_type_340 = convert_element_type_342 = permute_91 = None
        getitem_139 = _scaled_dot_product_flash_attention_9[0];  _scaled_dot_product_flash_attention_9 = None
        permute_92 = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_147 = torch.ops.aten.view.default(permute_92, [16, 513, 768]);  permute_92 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_344 = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        view_148 = torch.ops.aten.view.default(view_147, [8208, 768]);  view_147 = None
        permute_93 = torch.ops.aten.permute.default(convert_element_type_344, [1, 0]);  convert_element_type_344 = None
        addmm_53 = torch.ops.aten.addmm.default(convert_element_type_343, view_148, permute_93);  convert_element_type_343 = view_148 = permute_93 = None
        view_149 = torch.ops.aten.view.default(addmm_53, [16, 513, 768]);  addmm_53 = None
        add_65 = torch.ops.aten.add.Tensor(add_61, view_149);  add_61 = view_149 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_148 = var_mean_23[0]
        getitem_149 = var_mean_23[1];  var_mean_23 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_65, getitem_149);  getitem_149 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_35);  sub_23 = rsqrt_35 = None
        convert_element_type_348 = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_349 = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        convert_element_type_350 = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        view_150 = torch.ops.aten.view.default(convert_element_type_350, [8208, 768]);  convert_element_type_350 = None
        permute_94 = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_54 = torch.ops.aten.addmm.default(convert_element_type_348, view_150, permute_94);  convert_element_type_348 = view_150 = permute_94 = None
        view_151 = torch.ops.aten.view.default(addmm_54, [16, 513, 3072]);  addmm_54 = None
        convert_element_type_354 = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_91 = torch.ops.aten.mul.Tensor(convert_element_type_354, 0.5)
        mul_92 = torch.ops.aten.mul.Tensor(convert_element_type_354, 0.7071067811865476);  convert_element_type_354 = None
        erf_9 = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_67 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_91, add_67);  mul_91 = add_67 = None
        convert_element_type_355 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        convert_element_type_356 = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        view_152 = torch.ops.aten.view.default(convert_element_type_355, [8208, 3072]);  convert_element_type_355 = None
        permute_95 = torch.ops.aten.permute.default(convert_element_type_357, [1, 0]);  convert_element_type_357 = None
        addmm_55 = torch.ops.aten.addmm.default(convert_element_type_356, view_152, permute_95);  convert_element_type_356 = view_152 = permute_95 = None
        view_153 = torch.ops.aten.view.default(addmm_55, [16, 513, 768]);  addmm_55 = None
        add_68 = torch.ops.aten.add.Tensor(add_65, view_153);  add_65 = view_153 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_150 = var_mean_24[0]
        getitem_151 = var_mean_24[1];  var_mean_24 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_68, getitem_151);  getitem_151 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_36);  sub_24 = rsqrt_36 = None
        convert_element_type_361 = torch.ops.prims.convert_element_type.default(primals_130, torch.bfloat16);  primals_130 = None
        convert_element_type_362 = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        convert_element_type_363 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        view_154 = torch.ops.aten.view.default(convert_element_type_363, [8208, 768]);  convert_element_type_363 = None
        permute_96 = torch.ops.aten.permute.default(convert_element_type_362, [1, 0]);  convert_element_type_362 = None
        addmm_56 = torch.ops.aten.addmm.default(convert_element_type_361, view_154, permute_96);  convert_element_type_361 = permute_96 = None
        view_155 = torch.ops.aten.view.default(addmm_56, [16, 513, 1536]);  addmm_56 = None
        split_6 = torch.ops.aten.split.Tensor(view_155, 768, -1);  view_155 = None
        getitem_152 = split_6[0]
        getitem_153 = split_6[1];  split_6 = None
        convert_element_type_367 = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_368 = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        permute_97 = torch.ops.aten.permute.default(convert_element_type_368, [1, 0]);  convert_element_type_368 = None
        addmm_57 = torch.ops.aten.addmm.default(convert_element_type_367, view_154, permute_97);  convert_element_type_367 = view_154 = permute_97 = None
        view_157 = torch.ops.aten.view.default(addmm_57, [16, 513, 768]);  addmm_57 = None
        view_158 = torch.ops.aten.view.default(getitem_152, [16, 513, 12, -1]);  getitem_152 = None
        permute_98 = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        view_159 = torch.ops.aten.view.default(getitem_153, [16, 513, 12, -1]);  getitem_153 = None
        permute_99 = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        view_160 = torch.ops.aten.view.default(view_157, [16, 513, 12, -1]);  view_157 = None
        permute_100 = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        convert_element_type_373 = torch.ops.prims.convert_element_type.default(permute_98, torch.float32);  permute_98 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_373, convert_element_type_373)
        mean_12 = torch.ops.aten.mean.dim(mul_95, [-1], True);  mul_95 = None
        add_70 = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_373, rsqrt_37);  convert_element_type_373 = rsqrt_37 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, primals_133);  mul_96 = primals_133 = None
        convert_element_type_374 = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16);  mul_97 = None
        convert_element_type_375 = torch.ops.prims.convert_element_type.default(permute_99, torch.float32);  permute_99 = None
        mul_98 = torch.ops.aten.mul.Tensor(convert_element_type_375, convert_element_type_375)
        mean_13 = torch.ops.aten.mean.dim(mul_98, [-1], True);  mul_98 = None
        add_71 = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_99 = torch.ops.aten.mul.Tensor(convert_element_type_375, rsqrt_38);  convert_element_type_375 = rsqrt_38 = None
        mul_100 = torch.ops.aten.mul.Tensor(mul_99, primals_134);  mul_99 = primals_134 = None
        convert_element_type_376 = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None
        _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_374, convert_element_type_376, permute_100, scale = 0.125);  convert_element_type_374 = convert_element_type_376 = permute_100 = None
        getitem_154 = _scaled_dot_product_flash_attention_10[0];  _scaled_dot_product_flash_attention_10 = None
        permute_101 = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
        view_161 = torch.ops.aten.view.default(permute_101, [16, 513, 768]);  permute_101 = None
        convert_element_type_377 = torch.ops.prims.convert_element_type.default(primals_136, torch.bfloat16);  primals_136 = None
        convert_element_type_378 = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        view_162 = torch.ops.aten.view.default(view_161, [8208, 768]);  view_161 = None
        permute_102 = torch.ops.aten.permute.default(convert_element_type_378, [1, 0]);  convert_element_type_378 = None
        addmm_58 = torch.ops.aten.addmm.default(convert_element_type_377, view_162, permute_102);  convert_element_type_377 = view_162 = permute_102 = None
        view_163 = torch.ops.aten.view.default(addmm_58, [16, 513, 768]);  addmm_58 = None
        add_72 = torch.ops.aten.add.Tensor(add_68, view_163);  add_68 = view_163 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_163 = var_mean_25[0]
        getitem_164 = var_mean_25[1];  var_mean_25 = None
        add_73 = torch.ops.aten.add.Tensor(getitem_163, 1e-06);  getitem_163 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_72, getitem_164);  getitem_164 = None
        mul_101 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_39);  sub_25 = rsqrt_39 = None
        convert_element_type_382 = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_383 = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_384 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        view_164 = torch.ops.aten.view.default(convert_element_type_384, [8208, 768]);  convert_element_type_384 = None
        permute_103 = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        addmm_59 = torch.ops.aten.addmm.default(convert_element_type_382, view_164, permute_103);  convert_element_type_382 = view_164 = permute_103 = None
        view_165 = torch.ops.aten.view.default(addmm_59, [16, 513, 3072]);  addmm_59 = None
        convert_element_type_388 = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_388, 0.5)
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_388, 0.7071067811865476);  convert_element_type_388 = None
        erf_10 = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_74 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_102, add_74);  mul_102 = add_74 = None
        convert_element_type_389 = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        convert_element_type_390 = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        view_166 = torch.ops.aten.view.default(convert_element_type_389, [8208, 3072]);  convert_element_type_389 = None
        permute_104 = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_60 = torch.ops.aten.addmm.default(convert_element_type_390, view_166, permute_104);  convert_element_type_390 = view_166 = permute_104 = None
        view_167 = torch.ops.aten.view.default(addmm_60, [16, 513, 768]);  addmm_60 = None
        add_75 = torch.ops.aten.add.Tensor(add_72, view_167);  add_72 = view_167 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_165 = var_mean_26[0]
        getitem_166 = var_mean_26[1];  var_mean_26 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_75, getitem_166);  getitem_166 = None
        mul_105 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_40);  sub_26 = rsqrt_40 = None
        convert_element_type_395 = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_396 = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        view_168 = torch.ops.aten.view.default(convert_element_type_397, [8208, 768]);  convert_element_type_397 = None
        permute_105 = torch.ops.aten.permute.default(convert_element_type_396, [1, 0]);  convert_element_type_396 = None
        addmm_61 = torch.ops.aten.addmm.default(convert_element_type_395, view_168, permute_105);  convert_element_type_395 = permute_105 = None
        view_169 = torch.ops.aten.view.default(addmm_61, [16, 513, 1536]);  addmm_61 = None
        split_7 = torch.ops.aten.split.Tensor(view_169, 768, -1);  view_169 = None
        getitem_167 = split_7[0]
        getitem_168 = split_7[1];  split_7 = None
        convert_element_type_401 = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_402 = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        permute_106 = torch.ops.aten.permute.default(convert_element_type_402, [1, 0]);  convert_element_type_402 = None
        addmm_62 = torch.ops.aten.addmm.default(convert_element_type_401, view_168, permute_106);  convert_element_type_401 = view_168 = permute_106 = None
        view_171 = torch.ops.aten.view.default(addmm_62, [16, 513, 768]);  addmm_62 = None
        view_172 = torch.ops.aten.view.default(getitem_167, [16, 513, 12, -1]);  getitem_167 = None
        permute_107 = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        view_173 = torch.ops.aten.view.default(getitem_168, [16, 513, 12, -1]);  getitem_168 = None
        permute_108 = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        view_174 = torch.ops.aten.view.default(view_171, [16, 513, 12, -1]);  view_171 = None
        permute_109 = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        convert_element_type_407 = torch.ops.prims.convert_element_type.default(permute_107, torch.float32);  permute_107 = None
        mul_106 = torch.ops.aten.mul.Tensor(convert_element_type_407, convert_element_type_407)
        mean_14 = torch.ops.aten.mean.dim(mul_106, [-1], True);  mul_106 = None
        add_77 = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_107 = torch.ops.aten.mul.Tensor(convert_element_type_407, rsqrt_41);  convert_element_type_407 = rsqrt_41 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, primals_145);  mul_107 = primals_145 = None
        convert_element_type_408 = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(permute_108, torch.float32);  permute_108 = None
        mul_109 = torch.ops.aten.mul.Tensor(convert_element_type_409, convert_element_type_409)
        mean_15 = torch.ops.aten.mean.dim(mul_109, [-1], True);  mul_109 = None
        add_78 = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_409, rsqrt_42);  convert_element_type_409 = rsqrt_42 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, primals_146);  mul_110 = primals_146 = None
        convert_element_type_410 = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_408, convert_element_type_410, permute_109, scale = 0.125);  convert_element_type_408 = convert_element_type_410 = permute_109 = None
        getitem_169 = _scaled_dot_product_flash_attention_11[0];  _scaled_dot_product_flash_attention_11 = None
        permute_110 = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None
        view_175 = torch.ops.aten.view.default(permute_110, [16, 513, 768]);  permute_110 = None
        convert_element_type_411 = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_412 = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        view_176 = torch.ops.aten.view.default(view_175, [8208, 768]);  view_175 = None
        permute_111 = torch.ops.aten.permute.default(convert_element_type_412, [1, 0]);  convert_element_type_412 = None
        addmm_63 = torch.ops.aten.addmm.default(convert_element_type_411, view_176, permute_111);  convert_element_type_411 = view_176 = permute_111 = None
        view_177 = torch.ops.aten.view.default(addmm_63, [16, 513, 768]);  addmm_63 = None
        add_79 = torch.ops.aten.add.Tensor(add_75, view_177);  add_75 = view_177 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_178 = var_mean_27[0]
        getitem_179 = var_mean_27[1];  var_mean_27 = None
        add_80 = torch.ops.aten.add.Tensor(getitem_178, 1e-06);  getitem_178 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_27 = torch.ops.aten.sub.Tensor(add_79, getitem_179);  getitem_179 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_43);  sub_27 = rsqrt_43 = None
        convert_element_type_416 = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_417 = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        view_178 = torch.ops.aten.view.default(convert_element_type_418, [8208, 768]);  convert_element_type_418 = None
        permute_112 = torch.ops.aten.permute.default(convert_element_type_417, [1, 0]);  convert_element_type_417 = None
        addmm_64 = torch.ops.aten.addmm.default(convert_element_type_416, view_178, permute_112);  convert_element_type_416 = view_178 = permute_112 = None
        view_179 = torch.ops.aten.view.default(addmm_64, [16, 513, 3072]);  addmm_64 = None
        convert_element_type_422 = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        mul_113 = torch.ops.aten.mul.Tensor(convert_element_type_422, 0.5)
        mul_114 = torch.ops.aten.mul.Tensor(convert_element_type_422, 0.7071067811865476);  convert_element_type_422 = None
        erf_11 = torch.ops.aten.erf.default(mul_114);  mul_114 = None
        add_81 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_113, add_81);  mul_113 = add_81 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16);  mul_115 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convert_element_type_425 = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        view_180 = torch.ops.aten.view.default(convert_element_type_423, [8208, 3072]);  convert_element_type_423 = None
        permute_113 = torch.ops.aten.permute.default(convert_element_type_425, [1, 0]);  convert_element_type_425 = None
        addmm_65 = torch.ops.aten.addmm.default(convert_element_type_424, view_180, permute_113);  convert_element_type_424 = view_180 = permute_113 = None
        view_181 = torch.ops.aten.view.default(addmm_65, [16, 513, 768]);  addmm_65 = None
        add_82 = torch.ops.aten.add.Tensor(add_79, view_181);  add_79 = view_181 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
        getitem_180 = var_mean_28[0]
        getitem_181 = var_mean_28[1];  var_mean_28 = None
        add_83 = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_82, getitem_181);  getitem_181 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_44);  sub_28 = rsqrt_44 = None
        convert_element_type_429 = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        view_182 = torch.ops.aten.view.default(convert_element_type_431, [8208, 768]);  convert_element_type_431 = None
        permute_114 = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_66 = torch.ops.aten.addmm.default(convert_element_type_429, view_182, permute_114);  convert_element_type_429 = permute_114 = None
        view_183 = torch.ops.aten.view.default(addmm_66, [16, 513, 1536]);  addmm_66 = None
        split_8 = torch.ops.aten.split.Tensor(view_183, 768, -1);  view_183 = None
        getitem_182 = split_8[0]
        getitem_183 = split_8[1];  split_8 = None
        convert_element_type_435 = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        permute_115 = torch.ops.aten.permute.default(convert_element_type_436, [1, 0]);  convert_element_type_436 = None
        addmm_67 = torch.ops.aten.addmm.default(convert_element_type_435, view_182, permute_115);  convert_element_type_435 = view_182 = permute_115 = None
        view_185 = torch.ops.aten.view.default(addmm_67, [16, 513, 768]);  addmm_67 = None
        view_186 = torch.ops.aten.view.default(getitem_182, [16, 513, 12, -1]);  getitem_182 = None
        permute_116 = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        view_187 = torch.ops.aten.view.default(getitem_183, [16, 513, 12, -1]);  getitem_183 = None
        permute_117 = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        view_188 = torch.ops.aten.view.default(view_185, [16, 513, 12, -1]);  view_185 = None
        permute_118 = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        convert_element_type_441 = torch.ops.prims.convert_element_type.default(permute_116, torch.float32);  permute_116 = None
        mul_117 = torch.ops.aten.mul.Tensor(convert_element_type_441, convert_element_type_441)
        mean_16 = torch.ops.aten.mean.dim(mul_117, [-1], True);  mul_117 = None
        add_84 = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_441, rsqrt_45);  convert_element_type_441 = rsqrt_45 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, primals_157);  mul_118 = primals_157 = None
        convert_element_type_442 = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        convert_element_type_443 = torch.ops.prims.convert_element_type.default(permute_117, torch.float32);  permute_117 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_443, convert_element_type_443)
        mean_17 = torch.ops.aten.mean.dim(mul_120, [-1], True);  mul_120 = None
        add_85 = torch.ops.aten.add.Tensor(mean_17, 1e-05);  mean_17 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_121 = torch.ops.aten.mul.Tensor(convert_element_type_443, rsqrt_46);  convert_element_type_443 = rsqrt_46 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, primals_158);  mul_121 = primals_158 = None
        convert_element_type_444 = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        _scaled_dot_product_flash_attention_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(convert_element_type_442, convert_element_type_444, permute_118, scale = 0.125);  convert_element_type_442 = convert_element_type_444 = permute_118 = None
        getitem_184 = _scaled_dot_product_flash_attention_12[0];  _scaled_dot_product_flash_attention_12 = None
        permute_119 = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None
        view_189 = torch.ops.aten.view.default(permute_119, [16, 513, 768]);  permute_119 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        view_190 = torch.ops.aten.view.default(view_189, [8208, 768]);  view_189 = None
        permute_120 = torch.ops.aten.permute.default(convert_element_type_446, [1, 0]);  convert_element_type_446 = None
        addmm_68 = torch.ops.aten.addmm.default(convert_element_type_445, view_190, permute_120);  convert_element_type_445 = view_190 = permute_120 = None
        view_191 = torch.ops.aten.view.default(addmm_68, [16, 513, 768]);  addmm_68 = None
        add_86 = torch.ops.aten.add.Tensor(add_82, view_191);  add_82 = view_191 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_193 = var_mean_29[0]
        getitem_194 = var_mean_29[1];  var_mean_29 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_193, 1e-06);  getitem_193 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_29 = torch.ops.aten.sub.Tensor(add_86, getitem_194);  getitem_194 = None
        mul_123 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_47);  sub_29 = rsqrt_47 = None
        convert_element_type_450 = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_451 = torch.ops.prims.convert_element_type.default(primals_161, torch.bfloat16);  primals_161 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(mul_123, torch.bfloat16);  mul_123 = None
        view_192 = torch.ops.aten.view.default(convert_element_type_452, [8208, 768]);  convert_element_type_452 = None
        permute_121 = torch.ops.aten.permute.default(convert_element_type_451, [1, 0]);  convert_element_type_451 = None
        addmm_69 = torch.ops.aten.addmm.default(convert_element_type_450, view_192, permute_121);  convert_element_type_450 = view_192 = permute_121 = None
        view_193 = torch.ops.aten.view.default(addmm_69, [16, 513, 3072]);  addmm_69 = None
        convert_element_type_456 = torch.ops.prims.convert_element_type.default(view_193, torch.float32);  view_193 = None
        mul_124 = torch.ops.aten.mul.Tensor(convert_element_type_456, 0.5)
        mul_125 = torch.ops.aten.mul.Tensor(convert_element_type_456, 0.7071067811865476);  convert_element_type_456 = None
        erf_12 = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_88 = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_124, add_88);  mul_124 = add_88 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_458 = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_459 = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        view_194 = torch.ops.aten.view.default(convert_element_type_457, [8208, 3072]);  convert_element_type_457 = None
        permute_122 = torch.ops.aten.permute.default(convert_element_type_459, [1, 0]);  convert_element_type_459 = None
        addmm_70 = torch.ops.aten.addmm.default(convert_element_type_458, view_194, permute_122);  convert_element_type_458 = view_194 = permute_122 = None
        view_195 = torch.ops.aten.view.default(addmm_70, [16, 513, 768]);  addmm_70 = None
        add_89 = torch.ops.aten.add.Tensor(add_86, view_195);  add_86 = view_195 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_195 = var_mean_30[0]
        getitem_196 = var_mean_30[1];  var_mean_30 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_195, 1e-06);  getitem_195 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_89, getitem_196);  add_89 = getitem_196 = None
        mul_127 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_48);  sub_30 = rsqrt_48 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, primals_165);  mul_127 = primals_165 = None
        add_91 = torch.ops.aten.add.Tensor(mul_128, primals_166);  mul_128 = primals_166 = None
        slice_4 = torch.ops.aten.slice.Tensor(add_91, 1, 1, 9223372036854775807);  add_91 = None
        permute_123 = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        expand_1 = torch.ops.aten.expand.default(slice_4, [16, 512, 768]);  slice_4 = None
        expand_2 = torch.ops.aten.expand.default(permute_123, [16, 768, 32]);  permute_123 = None
        bmm = torch.ops.aten.bmm.default(expand_1, expand_2);  expand_1 = expand_2 = None
        add_92 = torch.ops.aten.add.Tensor(bmm, primals_168);  bmm = primals_168 = None
        mul_129 = torch.ops.aten.mul.Tensor(add_92, add_92)
        mean_18 = torch.ops.aten.mean.dim(mul_129, [-1], True);  mul_129 = None
        add_93 = torch.ops.aten.add.Tensor(mean_18, 1e-05);  mean_18 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_130 = torch.ops.aten.mul.Tensor(add_92, rsqrt_49);  add_92 = rsqrt_49 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, primals_169);  mul_130 = None
        mul_132 = torch.ops.aten.mul.Tensor(primals_170, primals_171);  primals_170 = primals_171 = None
        add_94 = torch.ops.aten.add.Tensor(mul_132, primals_172);  mul_132 = primals_172 = None
        mul_133 = torch.ops.aten.mul.Tensor(add_94, add_94)
        mean_19 = torch.ops.aten.mean.dim(mul_133, [-1], True);  mul_133 = None
        add_95 = torch.ops.aten.add.Tensor(mean_19, 1e-05);  mean_19 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_94, rsqrt_50);  add_94 = rsqrt_50 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, primals_169);  mul_134 = primals_169 = None
        view_199 = torch.ops.aten.view.default(mul_131, [-1, 32]);  mul_131 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_199, 2)
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [-1], True);  pow_1 = None
        full_default = torch.ops.aten.full.default([8192, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(mul_135, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_2, [-1], True);  pow_2 = None
        full_default_1 = torch.ops.aten.full.default([16384, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_136 = torch.ops.aten.mul.Tensor(view_199, -2);  view_199 = None
        cat_2 = torch.ops.aten.cat.default([mul_136, sum_1, full_default], -1);  mul_136 = sum_1 = full_default = None
        cat_3 = torch.ops.aten.cat.default([mul_135, full_default_1, sum_2], -1);  full_default_1 = sum_2 = None
        permute_124 = torch.ops.aten.permute.default(cat_3, [1, 0]);  cat_3 = None
        mm = torch.ops.aten.mm.default(cat_2, permute_124);  cat_2 = permute_124 = None
        clamp_min = torch.ops.aten.clamp_min.default(mm, 0);  mm = None
        sqrt = torch.ops.aten.sqrt.default(clamp_min);  clamp_min = None
        argmin = torch.ops.aten.argmin.default(sqrt, 1);  sqrt = None
        index = torch.ops.aten.index.Tensor(mul_135, [argmin]);  mul_135 = None
        view_203 = torch.ops.aten.view.default(index, [16, 512, -1]);  index = None
        view_204 = torch.ops.aten.view.default(argmin, [16, 512]);  argmin = None
        full_default_2 = torch.ops.aten.full.default([16, 1], 16384, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3 = torch.ops.aten.full.default([16, 1], 16385, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_4 = torch.ops.aten.cat.default([full_default_2, view_204, full_default_3], 1);  full_default_2 = full_default_3 = None
        device_put = torch.ops.prims.device_put.default(primals_175, device(type='cuda', index=0));  primals_175 = None
        device_put_1 = torch.ops.prims.device_put.default(primals_176, device(type='cuda', index=0));  primals_176 = None
        eq_2 = torch.ops.aten.eq.Scalar(device_put, 2)
        full_default_4 = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(device_put, [eq_2], full_default_4);  full_default_4 = None
        full_default_5 = torch.ops.aten.full.default([16, 1], 2, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_17 = torch.ops.aten.slice.Tensor(index_put, 1, 1, 9223372036854775807);  index_put = None
        cat_5 = torch.ops.aten.cat.default([slice_17, full_default_5], 1);  slice_17 = full_default_5 = None
        embedding = torch.ops.aten.embedding.default(primals_177, device_put, 2);  primals_177 = None
        convert_element_type_467 = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_468 = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(view_203, torch.bfloat16)
        view_208 = torch.ops.aten.view.default(convert_element_type_469, [8192, 32]);  convert_element_type_469 = None
        permute_126 = torch.ops.aten.permute.default(convert_element_type_468, [1, 0]);  convert_element_type_468 = None
        addmm_72 = torch.ops.aten.addmm.default(convert_element_type_467, view_208, permute_126);  convert_element_type_467 = permute_126 = None
        view_209 = torch.ops.aten.view.default(addmm_72, [16, 512, 2048])
        convert_element_type_473 = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_473, 0.5)
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_473, 0.7071067811865476);  convert_element_type_473 = None
        erf_13 = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_99 = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_144 = torch.ops.aten.mul.Tensor(mul_142, add_99);  mul_142 = add_99 = None
        convert_element_type_474 = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        convert_element_type_475 = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convert_element_type_476 = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        view_210 = torch.ops.aten.view.default(convert_element_type_474, [8192, 2048]);  convert_element_type_474 = None
        permute_127 = torch.ops.aten.permute.default(convert_element_type_476, [1, 0]);  convert_element_type_476 = None
        addmm_73 = torch.ops.aten.addmm.default(convert_element_type_475, view_210, permute_127);  convert_element_type_475 = None
        view_211 = torch.ops.aten.view.default(addmm_73, [16, 512, 2048]);  addmm_73 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(primals_182, 0);  primals_182 = None
        expand_5 = torch.ops.aten.expand.default(unsqueeze_3, [16, -1, -1]);  unsqueeze_3 = None
        add_100 = torch.ops.aten.add.Tensor(view_211, expand_5);  view_211 = expand_5 = None
        expand_6 = torch.ops.aten.expand.default(primals_183, [16, 1, -1]);  primals_183 = None
        expand_7 = torch.ops.aten.expand.default(primals_184, [16, 1, -1]);  primals_184 = None
        cat_6 = torch.ops.aten.cat.default([expand_6, add_100, expand_7], 1);  expand_6 = add_100 = expand_7 = None
        full_default_6 = torch.ops.aten.full.default([16, 514], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_7 = torch.ops.aten.cat.default([embedding, cat_6], 1);  embedding = cat_6 = None
        full_default_7 = torch.ops.aten.full.default([16, 642, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        uniform = torch.ops.aten.uniform.default(full_default_7, -0.0013081536848108147, 0.0013081536848108147);  full_default_7 = None
        add_101 = torch.ops.aten.add.Tensor(cat_7, uniform);  cat_7 = uniform = None
        native_dropout = torch.ops.aten.native_dropout.default(add_101, 0.1, True);  add_101 = None
        getitem_197 = native_dropout[0]
        getitem_198 = native_dropout[1];  native_dropout = None
        cat_8 = torch.ops.aten.cat.default([device_put_1, full_default_6], 1);  device_put_1 = full_default_6 = None
        cumsum_1 = torch.ops.aten.cumsum.default(cat_8, -1)
        sub_35 = torch.ops.aten.sub.Tensor(cumsum_1, 1);  cumsum_1 = None
        eq_3 = torch.ops.aten.eq.Scalar(cat_8, 0)
        full_default_8 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(eq_3, full_default_8, sub_35);  full_default_8 = sub_35 = None
        full_default_9 = torch.ops.aten.full.default([16, 514], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(where_1, full_default_9, 1, -514, 9223372036854775807);  where_1 = full_default_9 = None
        any_1 = torch.ops.aten.any.default(eq_3);  eq_3 = None
        # true_graph_0 = self.true_graph_0
        # false_graph_0 = self.false_graph_0
        # cond = torch.ops.higher_order.cond(any_1, true_graph_0, false_graph_0, [cat_8]);  any_1 = true_graph_0 = false_graph_0 = cat_8 = None
        # getitem_199 = cond[0];  cond = None
        permute_128 = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        # return (getitem_197, slice_scatter, getitem_199, cat_4, cat_5, view_204, view_203, device_put, eq_2, view_208, addmm_72, view_210, getitem_198, permute_128)
        return (getitem_197, slice_scatter, cat_4, cat_5, view_204, view_203, device_put, eq_2, view_208, addmm_72, view_210, getitem_198, permute_128)

def load_args(reader):
    buf0 = reader.storage('1724fbbe23f9f44c5e1be41ad216a08774fe4b2d', 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 8192, 6), is_leaf=True)  # primals_1
    buf1 = reader.storage('06db9624cd883292c6e9c597c66887cc01ff5f59', 1536, device=device(type='cuda', index=0))
    reader.tensor(buf1, (3, 128), is_leaf=True)  # primals_2
    buf2 = reader.storage('29b06030498db66f7f5edc20efe53a6ac6022449', 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # primals_3
    buf3 = reader.storage('1dec1db4b743cadc0cd0e29764bb0bbb97fa1e58', 1575936, device=device(type='cuda', index=0))
    reader.tensor(buf3, (513, 768), is_leaf=True)  # primals_4
    buf4 = reader.storage('dd3dc68bdebee3076806264cd7cfca616b653bf7', 2377728, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768, 774), is_leaf=True)  # primals_5
    buf5 = reader.storage('6411e1de0f88510bdbf71e37c12b76ec210b1250', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_6
    buf6 = reader.storage('f378202dc4fbfdde68c9dce4d7cd8e2d27bdbd57', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 768), is_leaf=True)  # primals_7
    buf7 = reader.storage('afd675afb046c8f998545d41fe76ffac39305305', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_8
    buf8 = reader.storage('ba86017eeab45274a6ebe97b59128a3faefaf998', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # primals_9
    buf9 = reader.storage('0e9c4340cd3605e961567f431eb427a2f2973c04', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # primals_10
    buf10 = reader.storage('842be50ffd01ed9f4ca00dbef880c22b17146bc1', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768, 768), is_leaf=True)  # primals_11
    buf11 = reader.storage('3d9a24ac7ecb85528e3e22b149fe88be2ba0978f', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # primals_12
    buf12 = reader.storage('6cb606b27883b7bdb884a52642d2ccdd8fd5ce36', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 768), is_leaf=True)  # primals_13
    buf13 = reader.storage('8f9f19878bf7e2d5dcb5460f5ab9eb9a2ad91ea7', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # primals_14
    buf14 = reader.storage('737709dcd622e6659d8c443a34701c8800ca9201', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768, 768), is_leaf=True)  # primals_15
    buf15 = reader.storage('616ebfa59845596c2abd52fc9ab178701b988196', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # primals_16
    buf16 = reader.storage('d7b98b208bcc809348d1f859358414d402e044d3', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf16, (3072, 768), is_leaf=True)  # primals_17
    buf17 = reader.storage('a3be72274013a6de6fdbe7943e505bef04966e44', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf17, (3072,), is_leaf=True)  # primals_18
    buf18 = reader.storage('33e080e10e2d3289efe650864fc3d91e9f64277d', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 3072), is_leaf=True)  # primals_19
    buf19 = reader.storage('8007ace52ab498ed731b4f2f474481968b7189cc', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # primals_20
    buf20 = reader.storage('f3341310dd725eecfaaebd14795ed93e7d60eafc', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1536, 768), is_leaf=True)  # primals_21
    buf21 = reader.storage('32eb4beae2b4d8f4c1c01b285832f2ae8ccd1d63', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1536,), is_leaf=True)  # primals_22
    buf22 = reader.storage('ce5459485671f641d8141ae7fb281f1668dd4424', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768, 768), is_leaf=True)  # primals_23
    buf23 = reader.storage('743b3a50ad94ba51a047d6e43578c241d15972f3', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_24
    buf24 = reader.storage('9a84e7ec13bd9420632683540230fc77b99c8332', 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # primals_25
    buf25 = reader.storage('981ff75a47e48b6b0a339c168e59c7372633a5a9', 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # primals_26
    buf26 = reader.storage('948e8743dd77d24d14949e2a38acf8562d32ecf5', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768, 768), is_leaf=True)  # primals_27
    buf27 = reader.storage('d6ae27e3d409e826a8fce807c320c6e530320c53', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # primals_28
    buf28 = reader.storage('baba0bc0bd6d5b6e6621eb5082088f4d71cf042d', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf28, (3072, 768), is_leaf=True)  # primals_29
    buf29 = reader.storage('b8eca65171b8bdd0dcc413169819b97bbd8ec26a', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf29, (3072,), is_leaf=True)  # primals_30
    buf30 = reader.storage('27c6de2a099e9fae3cd0312a8eae0722f6109ef3', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 3072), is_leaf=True)  # primals_31
    buf31 = reader.storage('2acf591993ccf5f923daf38deba6f3bb6525d4cb', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # primals_32
    buf32 = reader.storage('ebad2074cfb360991a87693a364aa53b92d2dfff', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 768), is_leaf=True)  # primals_33
    buf33 = reader.storage('13a39ddf0f2ccde1c7ec113a5a5b8268ad0c2227', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # primals_34
    buf34 = reader.storage('589b44ad5ba0747c4098198c5a4a2f4a84aec857', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768, 768), is_leaf=True)  # primals_35
    buf35 = reader.storage('ec9a57474229d8a3c4ab5259f9147145a47c1d79', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # primals_36
    buf36 = reader.storage('cbbeb9653d456175661185c3365316e7a9de41ce', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 768), is_leaf=True)  # primals_37
    buf37 = reader.storage('a3f127fffdac438118e2d455aabf0536ab77281c', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # primals_38
    buf38 = reader.storage('4231627dfbd0f8ac88a73a0c418bb13c6e1e994c', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768, 768), is_leaf=True)  # primals_39
    buf39 = reader.storage('fd47a2a6dd5f0490e5d8ff60e3a64ac91e2d6714', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # primals_40
    buf40 = reader.storage('ec35bb8bf4a3005d7e966ebed0946ddfe951ec26', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf40, (3072, 768), is_leaf=True)  # primals_41
    buf41 = reader.storage('36090c0d307283158e508dfa5472cdc49f2c84b4', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf41, (3072,), is_leaf=True)  # primals_42
    buf42 = reader.storage('933002089f920ff0a0a50f6b84f2aa1b5a68b3af', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 3072), is_leaf=True)  # primals_43
    buf43 = reader.storage('ebc1ab7b6274829291066735691682c916f4b1a1', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # primals_44
    buf44 = reader.storage('5b447ecdcb701799b6433c2426b4abb554cc3cfe', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1536, 768), is_leaf=True)  # primals_45
    buf45 = reader.storage('518a39e76dcd4279f21bd431c375fddfabceffe4', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1536,), is_leaf=True)  # primals_46
    buf46 = reader.storage('09288b5c0bf2df84f2857c34728a5d3de78e57b8', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768, 768), is_leaf=True)  # primals_47
    buf47 = reader.storage('202c8cceb4278854e961da35fb71b6541810d759', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # primals_48
    buf48 = reader.storage('fbdf78ee140e9e051adcab5f9c1d6a693280e9af', 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # primals_49
    buf49 = reader.storage('6099d9d334254f2b71f1b4b76b8ebdbcee5fa7a2', 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # primals_50
    buf50 = reader.storage('d9df513b63183427ab309ec336bf87e13fe042ad', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768, 768), is_leaf=True)  # primals_51
    buf51 = reader.storage('8adbd9599060f3be3ca8e9724031238ab3d11b4d', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # primals_52
    buf52 = reader.storage('83ff3b36b9311779b3c900c57a5e8a1f64d25aab', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3072, 768), is_leaf=True)  # primals_53
    buf53 = reader.storage('b1d05a866643104acd8d71eb56bf66b0f4b32d62', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (3072,), is_leaf=True)  # primals_54
    buf54 = reader.storage('86c77baab5c5d1dbae1a0a9388210b6295cbb70d', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 3072), is_leaf=True)  # primals_55
    buf55 = reader.storage('8f67af03ec730feb50134f8cd8ffc3538ad373f1', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # primals_56
    buf56 = reader.storage('8631a7ba1c6a8967bd3a9d41ca81efcc2999c618', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # primals_57
    buf57 = reader.storage('76929d14c5e5e504377b67fe8700d43009261b8d', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # primals_58
    buf58 = reader.storage('63b81dade2dc1117500944bb17b9cc47d292b033', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768, 768), is_leaf=True)  # primals_59
    buf59 = reader.storage('3b5eeea1e4d7c22a40bbcdd5b774c6b1250804ca', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # primals_60
    buf60 = reader.storage('6ff1c209efa5ba0d3bd712d8826e44b2e4660a7e', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 768), is_leaf=True)  # primals_61
    buf61 = reader.storage('5c6e56ab695ffac28ae2f19006eb9a7a1dcab7c7', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # primals_62
    buf62 = reader.storage('6eab638806871a1b74a17612b4bd7155f60c632f', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768, 768), is_leaf=True)  # primals_63
    buf63 = reader.storage('92d1b957ff91befc9ee0aabea216936fc632c50c', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # primals_64
    buf64 = reader.storage('d410f9878236f60acace906fa96cb53d016a1beb', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf64, (3072, 768), is_leaf=True)  # primals_65
    buf65 = reader.storage('c26645f02a6eb1d45cf81da79fa3b07951d1ddb1', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf65, (3072,), is_leaf=True)  # primals_66
    buf66 = reader.storage('10b5ed26705b3d29aea4d4e6230b854be60d59e6', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 3072), is_leaf=True)  # primals_67
    buf67 = reader.storage('cfaca915e98e0842ab0562ae49f1795d0010d950', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # primals_68
    buf68 = reader.storage('54f859438071381128ba047028dc5beca4079d78', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1536, 768), is_leaf=True)  # primals_69
    buf69 = reader.storage('afdcef22eaf20abfa1871d3519b354b7c5f27c83', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1536,), is_leaf=True)  # primals_70
    buf70 = reader.storage('0af23b8b21843b97a061db7c7ad690f6ace3c449', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768, 768), is_leaf=True)  # primals_71
    buf71 = reader.storage('74a3508b880b336b184377b0b543aca22261aad5', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # primals_72
    buf72 = reader.storage('27d461d967c5bb83fde1a6c9792d11595ba516f5', 256, device=device(type='cuda', index=0))
    reader.tensor(buf72, (64,), is_leaf=True)  # primals_73
    buf73 = reader.storage('380e7be9c10a4d752e1ddfce3ea11bc15d4c98ef', 256, device=device(type='cuda', index=0))
    reader.tensor(buf73, (64,), is_leaf=True)  # primals_74
    buf74 = reader.storage('848528b352589169a113e87a6a7840c63a4e8932', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 768), is_leaf=True)  # primals_75
    buf75 = reader.storage('6fa130ce73ad1a54edd66ad67061ab09fdb57942', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # primals_76
    buf76 = reader.storage('3bb6fce95a6c2481330a692a00f189bc386f6202', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf76, (3072, 768), is_leaf=True)  # primals_77
    buf77 = reader.storage('ca939749100f3cf29e554367df00d13c42a47bec', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf77, (3072,), is_leaf=True)  # primals_78
    buf78 = reader.storage('00fee103f70a148f49e84bf248b4fd0ea90ff85e', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 3072), is_leaf=True)  # primals_79
    buf79 = reader.storage('9278a67aa1e94dca22c8379cc143d07546e57e73', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # primals_80
    buf80 = reader.storage('e6e49fa1605c68628634e196f9b0c24730cd9ff5', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf80, (1536, 768), is_leaf=True)  # primals_81
    buf81 = reader.storage('6e3614acdeea3ae4124575e08fd71c86a8b86c7c', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf81, (1536,), is_leaf=True)  # primals_82
    buf82 = reader.storage('fe58f8dc856b2cfb7815dce2f418d8b8f325a4c8', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768, 768), is_leaf=True)  # primals_83
    buf83 = reader.storage('56ef7ed2cbb95938fa9ebea628e47bce603c1eec', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # primals_84
    buf84 = reader.storage('c132809695f23cb8c4bd90ce624b2fd8755ead65', 256, device=device(type='cuda', index=0))
    reader.tensor(buf84, (64,), is_leaf=True)  # primals_85
    buf85 = reader.storage('fad993eaa9b5ae0a0c0894c25cb4bbc996261ab1', 256, device=device(type='cuda', index=0))
    reader.tensor(buf85, (64,), is_leaf=True)  # primals_86
    buf86 = reader.storage('cba3fa1670c5cf257c646986bc93811116c9ec9c', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768, 768), is_leaf=True)  # primals_87
    buf87 = reader.storage('e538cd8e1fd946d0c89cb7efa1a11d4272767200', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_88
    buf88 = reader.storage('fa708b4fab065376bc4ec7eb2a6ec831c7b1c6e9', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf88, (3072, 768), is_leaf=True)  # primals_89
    buf89 = reader.storage('8b8cf305a13fbf83cd1ec7a462eb6656bb6bb7f2', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf89, (3072,), is_leaf=True)  # primals_90
    buf90 = reader.storage('f27326a3518935f2a3333194a1a35755e166c599', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 3072), is_leaf=True)  # primals_91
    buf91 = reader.storage('6d7290996196276f14c8acacd5ce1da2296ee8c8', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # primals_92
    buf92 = reader.storage('654ef32cc83af8798322d900bb6c6a23c77fc090', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf92, (1536, 768), is_leaf=True)  # primals_93
    buf93 = reader.storage('3526112138252ef4968ea2579758e19e0011c2c4', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf93, (1536,), is_leaf=True)  # primals_94
    buf94 = reader.storage('b28d1d488a63815a68fcd59645eef836127ea2ba', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 768), is_leaf=True)  # primals_95
    buf95 = reader.storage('33a71adaff171fe890a25494440b577fcb7cf576', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # primals_96
    buf96 = reader.storage('e887f6737c35cea30346e8abbd6f78785ed19749', 256, device=device(type='cuda', index=0))
    reader.tensor(buf96, (64,), is_leaf=True)  # primals_97
    buf97 = reader.storage('241769a870f10dc47903506fe2dfb589ab620177', 256, device=device(type='cuda', index=0))
    reader.tensor(buf97, (64,), is_leaf=True)  # primals_98
    buf98 = reader.storage('40246720255cac2cd2b9592c870e00f61d37eefa', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768, 768), is_leaf=True)  # primals_99
    buf99 = reader.storage('43f957a6562156a925e3bb52f407715846dba574', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # primals_100
    buf100 = reader.storage('c0ab92294dbaf950a60557e61a7cb4cb47e0dec0', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf100, (3072, 768), is_leaf=True)  # primals_101
    buf101 = reader.storage('fbad8da42f48ecf6b6efa836216410c71f054638', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf101, (3072,), is_leaf=True)  # primals_102
    buf102 = reader.storage('16c45e1931d6319928586c93f885816e3114b28d', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 3072), is_leaf=True)  # primals_103
    buf103 = reader.storage('0eb5056645d22a495b52c22e983b8c188cb27e46', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # primals_104
    buf104 = reader.storage('4b1074e95fd285e283f2f4b309caf3c1e4d14447', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768, 768), is_leaf=True)  # primals_105
    buf105 = reader.storage('649bd7eff0886de4b6bbf2192d0f7324823b943d', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # primals_106
    buf106 = reader.storage('ae2bce1064953c268dc8e9d8ff844af7bf166bfd', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768, 768), is_leaf=True)  # primals_107
    buf107 = reader.storage('a5081aa79d5e5bf1a734c134ccf89e1c2acb1878', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # primals_108
    buf108 = reader.storage('c48d9bea88fd2f16494469747374f9ac358c1cdb', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 768), is_leaf=True)  # primals_109
    buf109 = reader.storage('705baefb3148cd213c8f174d5aa864e729c9dea7', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # primals_110
    buf110 = reader.storage('918966db8aff33b2650eae6cb6bcd6128e8846dd', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 768), is_leaf=True)  # primals_111
    buf111 = reader.storage('270bcac04dea7b80c0759d7058bb33313d37b028', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # primals_112
    buf112 = reader.storage('5882c402af796085e04a38aebe2babe392f0da1d', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf112, (3072, 768), is_leaf=True)  # primals_113
    buf113 = reader.storage('2ac2e0ae8a1a2cf37b57373bdded7dbb163b2054', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf113, (3072,), is_leaf=True)  # primals_114
    buf114 = reader.storage('3b35c1d928112c5f1bbebb8feb465abbe51331c6', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 3072), is_leaf=True)  # primals_115
    buf115 = reader.storage('27f9fbbafea033648f87068e9506167eb2120005', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # primals_116
    buf116 = reader.storage('046dda2eaea977cd71c37b2122626939abfb5f2a', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536, 768), is_leaf=True)  # primals_117
    buf117 = reader.storage('9cc288b3548a206c6895bdfac9ff1a312e034dc2', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1536,), is_leaf=True)  # primals_118
    buf118 = reader.storage('e05c78f4581e8be669c82abe4b82b361f0d47284', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768, 768), is_leaf=True)  # primals_119
    buf119 = reader.storage('3444a8cc41fcb92a70c4bf171c528ebcefcb300b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # primals_120
    buf120 = reader.storage('ad472a5bf30ac4803d872a0ca12210b3e945232f', 256, device=device(type='cuda', index=0))
    reader.tensor(buf120, (64,), is_leaf=True)  # primals_121
    buf121 = reader.storage('a190a8b588f1ddeece671f2f18947d095e2c4559', 256, device=device(type='cuda', index=0))
    reader.tensor(buf121, (64,), is_leaf=True)  # primals_122
    buf122 = reader.storage('38b75788bbdee96d7d24b1c89585f0200086343f', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768, 768), is_leaf=True)  # primals_123
    buf123 = reader.storage('922ebff56e9f7f8d89a0dec9979e2d7e18b7c75f', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # primals_124
    buf124 = reader.storage('1a307704ea881f561b02c6a06151b3ba90d09523', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf124, (3072, 768), is_leaf=True)  # primals_125
    buf125 = reader.storage('b9a62ef546bbc2022165ae85d4a56bd9ee34f83b', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf125, (3072,), is_leaf=True)  # primals_126
    buf126 = reader.storage('5c8813bd683c37f880d2d103a2992db3aa93595a', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 3072), is_leaf=True)  # primals_127
    buf127 = reader.storage('b4db33121153c6c28834ca7627b5d17f5142a47b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # primals_128
    buf128 = reader.storage('d161d11a64592c06bc16274ef04097a0ebee4276', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf128, (1536, 768), is_leaf=True)  # primals_129
    buf129 = reader.storage('6ad9f7f4dcddcc39d7a48ba845bd9499f350f25e', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf129, (1536,), is_leaf=True)  # primals_130
    buf130 = reader.storage('7ae43b3337b055d0688e18fc0a10e1cbd7103076', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768, 768), is_leaf=True)  # primals_131
    buf131 = reader.storage('bf2fd8de4940ee441f63cc8844131b14bb1e2d04', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # primals_132
    buf132 = reader.storage('735b4c59b2aadce5f650908cefde7cf0779514d1', 256, device=device(type='cuda', index=0))
    reader.tensor(buf132, (64,), is_leaf=True)  # primals_133
    buf133 = reader.storage('1d95126fae418b13f5bc592bc5184967194c52ac', 256, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64,), is_leaf=True)  # primals_134
    buf134 = reader.storage('d41f0d9fa036b0f2b1f889d46d22496857ee5f25', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768, 768), is_leaf=True)  # primals_135
    buf135 = reader.storage('c3e45582767c9c8a435ff3e3d7ea69ff9e89c158', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # primals_136
    buf136 = reader.storage('ee774d909ccad858b3cc157255fa403b76a24007', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf136, (3072, 768), is_leaf=True)  # primals_137
    buf137 = reader.storage('63b3d59ab8a7908fbc89bf892889be07e80a0ddd', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf137, (3072,), is_leaf=True)  # primals_138
    buf138 = reader.storage('f45617dc4c6cc8cee1b68787d31d2e9a46e6d71a', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 3072), is_leaf=True)  # primals_139
    buf139 = reader.storage('54ff187f051c65e7f87b6785dd8695658b0fb3b0', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # primals_140
    buf140 = reader.storage('decc6c5e5f8730cf6e9a3c4bbe28cc108d085faf', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1536, 768), is_leaf=True)  # primals_141
    buf141 = reader.storage('9213afe76baeaf0d1f2775cd1387b1567831a197', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1536,), is_leaf=True)  # primals_142
    buf142 = reader.storage('a1f2de29ec9baaf153e7734921d812023d5a076f', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 768), is_leaf=True)  # primals_143
    buf143 = reader.storage('acac55a366e738c5da4fdfffab8a1f301f6a31cc', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # primals_144
    buf144 = reader.storage('699db90d0bed3c72317fb5a16095f100327ae637', 256, device=device(type='cuda', index=0))
    reader.tensor(buf144, (64,), is_leaf=True)  # primals_145
    buf145 = reader.storage('37765c1253f33ed1e75a447c80fda6f8bc3bb392', 256, device=device(type='cuda', index=0))
    reader.tensor(buf145, (64,), is_leaf=True)  # primals_146
    buf146 = reader.storage('4694777b022fdead59db6e845e6576cbd2004da6', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768, 768), is_leaf=True)  # primals_147
    buf147 = reader.storage('b650f774c978c4b3157e6204c94eb43d9e4fbb7a', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # primals_148
    buf148 = reader.storage('cd694917fffc1d40ce4cbaa3f4f9f3466a2b4be8', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf148, (3072, 768), is_leaf=True)  # primals_149
    buf149 = reader.storage('77149cc327e09c14c956e5886c00f5b692456383', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf149, (3072,), is_leaf=True)  # primals_150
    buf150 = reader.storage('6a58919b434b5ac9ff8f349b7e80ff2cd12a4909', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768, 3072), is_leaf=True)  # primals_151
    buf151 = reader.storage('c57d2838ef2231a8d1cf1cb36c0dec9cf77f3380', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # primals_152
    buf152 = reader.storage('e2c04103b7763958ed40b7225983e01f3707685c', 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1536, 768), is_leaf=True)  # primals_153
    buf153 = reader.storage('1fa8f4c843193a010730b7a07cc1d48064303a36', 6144, device=device(type='cuda', index=0))
    reader.tensor(buf153, (1536,), is_leaf=True)  # primals_154
    buf154 = reader.storage('2702623fbf96021c65cba9e4749c9397a47f9acb', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768, 768), is_leaf=True)  # primals_155
    buf155 = reader.storage('a272c5166217c6edd5a49ab68a825ea3f516f03b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # primals_156
    buf156 = reader.storage('77791cc84043bb6dd77a4c378e4afde16d4e3be7', 256, device=device(type='cuda', index=0))
    reader.tensor(buf156, (64,), is_leaf=True)  # primals_157
    buf157 = reader.storage('6a693a244a758807bc5c2a9ddc7435ab40df9109', 256, device=device(type='cuda', index=0))
    reader.tensor(buf157, (64,), is_leaf=True)  # primals_158
    buf158 = reader.storage('54b067db262a1050c3c74914570d1488de7da375', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768, 768), is_leaf=True)  # primals_159
    buf159 = reader.storage('d72a3ede4312e6aa506129f18c50677522de92df', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768,), is_leaf=True)  # primals_160
    buf160 = reader.storage('3e0c79d4ab78ba44cab4eb43ee6e37775beda680', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf160, (3072, 768), is_leaf=True)  # primals_161
    buf161 = reader.storage('13ee56c8be8871ff2225c50fc91229604eecba9a', 12288, device=device(type='cuda', index=0))
    reader.tensor(buf161, (3072,), is_leaf=True)  # primals_162
    buf162 = reader.storage('089199cccb02ef9ea667e8038d8e11a96bb559b4', 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768, 3072), is_leaf=True)  # primals_163
    buf163 = reader.storage('622a0db2865958462d55b54de6b9a587de87984b', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # primals_164
    buf164 = reader.storage('2a190fd487ea403ea9d6212c272f349222d17f6e', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768,), is_leaf=True)  # primals_165
    buf165 = reader.storage('e4af785564a4e23b69eb93c0902c761f4bcaf111', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768,), is_leaf=True)  # primals_166
    buf166 = reader.storage('7e52472ed918983e1e56c860a42604b73fca5f95', 98304, device=device(type='cuda', index=0))
    reader.tensor(buf166, (32, 768), is_leaf=True)  # primals_167
    buf167 = reader.storage('4d98fbc9517105e5f2773e26d2187044cab9b0b6', 128, device=device(type='cuda', index=0))
    reader.tensor(buf167, (32,), is_leaf=True)  # primals_168
    buf168 = reader.storage('752a25d74555724ffd0e9026b53b1309ce7f8062', 128, device=device(type='cuda', index=0))
    reader.tensor(buf168, (32,), is_leaf=True)  # primals_169
    buf169 = reader.storage('e35855e4c5038aadd28d45d1b1d2c2f2e6fb1bd2', 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf169, (16384, 32), is_leaf=True)  # primals_170
    buf170 = reader.storage('cb83fbbf01f056b748c9d74c642d61551bb28e37', 128, device=device(type='cuda', index=0))
    reader.tensor(buf170, (32,), is_leaf=True)  # primals_171
    buf171 = reader.storage('66f567616946702d22c3103cacf73aed621bafb3', 128, device=device(type='cuda', index=0))
    reader.tensor(buf171, (32,), is_leaf=True)  # primals_172
    buf172 = reader.storage('39c818c92b8d8746cd82548b479de77edfe8cf55', 98304, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768, 32), is_leaf=True)  # primals_173
    buf173 = reader.storage('fe225c4a307f648c11a1e5ea9b2563762b1677de', 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), is_leaf=True)  # primals_174
    buf174 = reader.storage('7f444017d6e2235c466c307d43386d03bc37d1af', 16384, dtype_hint=torch.int64)
    reader.tensor(buf174, (16, 128), dtype=torch.int64, is_leaf=True)  # primals_175
    buf175 = reader.storage('53f313b37d9ae200ff4b2adce5d03629da347c07', 16384, dtype_hint=torch.int64)
    reader.tensor(buf175, (16, 128), dtype=torch.int64, is_leaf=True)  # primals_176
    buf176 = reader.storage('481b55ef5938e42347ab09fc335c561ac6458c8f', 758194176, device=device(type='cuda', index=0))
    reader.tensor(buf176, (92553, 2048), requires_grad=True, is_leaf=True)  # primals_177
    buf177 = reader.storage('3febe8847a00a66d02938cf8ccae9e2f856d21e7', 262144, device=device(type='cuda', index=0))
    reader.tensor(buf177, (2048, 32), requires_grad=True, is_leaf=True)  # primals_178
    buf178 = reader.storage('c46854ee4454f3409ae199b1dc5ce7377e323564', 8192, device=device(type='cuda', index=0))
    reader.tensor(buf178, (2048,), requires_grad=True, is_leaf=True)  # primals_179
    buf179 = reader.storage('cf3f2cc4bd92118e421309ba4cb67015c663d25e', 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf179, (2048, 2048), requires_grad=True, is_leaf=True)  # primals_180
    buf180 = reader.storage('55bfd913a6b886fa82efe9c6272f3fd11c7ac305', 8192, device=device(type='cuda', index=0))
    reader.tensor(buf180, (2048,), requires_grad=True, is_leaf=True)  # primals_181
    buf181 = reader.storage('b3a38688111504a3c79dea264b97f0989ca271d8', 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512, 2048), requires_grad=True, is_leaf=True)  # primals_182
    buf182 = reader.storage('cc0bf99d4513592c36a616c4677d5c18d39094f4', 8192, device=device(type='cuda', index=0))
    reader.tensor(buf182, (1, 1, 2048), requires_grad=True, is_leaf=True)  # primals_183
    buf183 = reader.storage('1d39e006082afd5a34f09b4035a367b430bf43c8', 8192, device=device(type='cuda', index=0))
    reader.tensor(buf183, (1, 1, 2048), requires_grad=True, is_leaf=True)  # primals_184
load_args._version = 0
mod = Repro()
# if __name__ == '__main__':
#     from torch._dynamo.repro.after_aot import run_repro
#     with torch.no_grad():
#        run_repro(mod, load_args, accuracy=True, command='run', save_dir='/tmp/slarsen/minifier/checkpoints', tracing_mode='real', check_str=None)
#        # To run it separately, do
#        mod, args = run_repro(mod, load_args, accuracy=True, command='get_args', save_dir='/tmp/slarsen/minifier/checkpoints', tracing_mode='real', check_str=None)
#        mod(*args)


from torch._dynamo.repro.after_aot import run_repro
from torch._inductor.compiler_bisector import CompilerBisector

_, args = run_repro(mod, load_args, accuracy=True, command='get_args', save_dir='/tmp/slarsen/minifier/checkpoints', tracing_mode='real', check_str=None)

def bisect_fn():
    torch._dynamo.reset()
    with torch.no_grad():
        mod1 = Repro()
        mod2 = torch.compile(Repro())
        torch.cuda.manual_seed(42)
        res1 = mod1(*args)
        torch.cuda.manual_seed(42)
        res2 = mod2(*args)
        return all(torch.allclose(a, b) for a, b in zip(res1, res2))

print(CompilerBisector.do_bisect(bisect_fn))