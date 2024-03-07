#date: 2024-03-07T18:39:11Z
#url: https://api.github.com/gists/f5f2b9201f9522b8d401baa742af6878
#owner: https://api.github.com/users/yf225

TRACED GRAPH
 ===== Forward graph 4 =====
 /data/users/willfeng/pytorch_yf225/torch/fx/_lazy_graph_module.py class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[2, 12340]", arg2_1: "f32[12340, 12340]", arg3_1: "f32[2, 12340]", arg4_1: "f32[12340]", arg5_1: "f32[12340]", arg6_1: "f32[12340, 12340]", arg7_1: "f32[76137800]", arg8_1: "f32[6170]", arg9_1: "f32[76137800]", arg10_1: "f32[6170]"):
        # File: <eval_with_key>.96:12 in forward, code: expand = torch.ops.aten.expand.default(getitem, [2, 12340]);  getitem = None
        expand: "f32[2, 12340]" = torch.ops.aten.expand.default(arg0_1, [2, 12340]);  arg0_1 = None
        
        # File: <eval_with_key>.96:14 in forward, code: clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        clone: "f32[2, 12340]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        
        # File: <eval_with_key>.96:15 in forward, code: empty = torch.ops.aten.empty.memory_format([304575880], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        empty: "f32[304575880]" = torch.ops.aten.empty.memory_format([304575880], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: <eval_with_key>.96:16 in forward, code: slice_1 = torch.ops.aten.slice.Tensor(empty, 0, 0, 152287940)
        slice_1: "f32[152287940]" = torch.ops.aten.slice.Tensor(empty, 0, 0, 152287940)
        
        # File: <eval_with_key>.96:17 in forward, code: split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [76137800, 6170, 76137800, 6170])
        split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [76137800, 6170, 76137800, 6170])
        getitem: "f32[76137800]" = split_with_sizes[0]
        getitem_1: "f32[6170]" = split_with_sizes[1]
        getitem_2: "f32[76137800]" = split_with_sizes[2]
        getitem_3: "f32[6170]" = split_with_sizes[3];  split_with_sizes = None
        
        # File: <eval_with_key>.96:22 in forward, code: slice_scatter = torch.ops.aten.slice_scatter.default(slice_1, getitem_8, 0, 0, 76137800);  slice_1 = getitem_8 = None
        slice_scatter: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_1, getitem, 0, 0, 76137800);  slice_1 = getitem = None
        
        # File: <eval_with_key>.96:23 in forward, code: slice_scatter_1 = torch.ops.aten.slice_scatter.default(empty, slice_scatter, 0, 0, 152287940);  empty = slice_scatter = None
        slice_scatter_1: "f32[304575880]" = torch.ops.aten.slice_scatter.default(empty, slice_scatter, 0, 0, 152287940);  empty = slice_scatter = None
        
        # File: <eval_with_key>.96:24 in forward, code: slice_2 = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 152287940)
        slice_2: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 152287940)
        
        # File: <eval_with_key>.96:25 in forward, code: slice_scatter_2 = torch.ops.aten.slice_scatter.default(slice_2, getitem_9, 0, 76137800, 76143970);  slice_2 = getitem_9 = None
        slice_scatter_2: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_2, getitem_1, 0, 76137800, 76143970);  slice_2 = getitem_1 = None
        
        # File: <eval_with_key>.96:26 in forward, code: slice_scatter_3 = torch.ops.aten.slice_scatter.default(slice_scatter_1, slice_scatter_2, 0, 0, 152287940);  slice_scatter_1 = slice_scatter_2 = None
        slice_scatter_3: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, slice_scatter_2, 0, 0, 152287940);  slice_scatter_1 = slice_scatter_2 = None
        
        # File: <eval_with_key>.96:27 in forward, code: slice_3 = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 152287940)
        slice_3: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 152287940)
        
        # File: <eval_with_key>.96:28 in forward, code: slice_scatter_4 = torch.ops.aten.slice_scatter.default(slice_3, getitem_10, 0, 76143970, 152281770);  slice_3 = getitem_10 = None
        slice_scatter_4: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_3, getitem_2, 0, 76143970, 152281770);  slice_3 = getitem_2 = None
        
        # File: <eval_with_key>.96:29 in forward, code: slice_scatter_5 = torch.ops.aten.slice_scatter.default(slice_scatter_3, slice_scatter_4, 0, 0, 152287940);  slice_scatter_3 = slice_scatter_4 = None
        slice_scatter_5: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, slice_scatter_4, 0, 0, 152287940);  slice_scatter_3 = slice_scatter_4 = None
        
        # File: <eval_with_key>.96:30 in forward, code: slice_4 = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 152287940)
        slice_4: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 152287940)
        
        # File: <eval_with_key>.96:31 in forward, code: slice_scatter_6 = torch.ops.aten.slice_scatter.default(slice_4, getitem_11, 0, 152281770, 152287940);  slice_4 = getitem_11 = None
        slice_scatter_6: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_4, getitem_3, 0, 152281770, 152287940);  slice_4 = getitem_3 = None
        
        # File: <eval_with_key>.96:32 in forward, code: slice_scatter_7 = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_6, 0, 0, 152287940);  slice_scatter_5 = slice_scatter_6 = None
        slice_scatter_7: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_6, 0, 0, 152287940);  slice_scatter_5 = slice_scatter_6 = None
        
        # File: <eval_with_key>.96:33 in forward, code: slice_5 = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 0, 152287940)
        slice_5: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 0, 152287940)
        
        # File: <eval_with_key>.96:34 in forward, code: all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(slice_5, 2, '0');  slice_5 = None
        all_gather_into_tensor: "f32[304575880]" = torch.ops._c10d_functional.all_gather_into_tensor.default(slice_5, 2, '0');  slice_5 = None
        
        # File: <eval_with_key>.96:35 in forward, code: wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        wait_tensor: "f32[304575880]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        
        # File: <eval_with_key>.96:36 in forward, code: copy = torch.ops.aten.copy.default(slice_scatter_7, wait_tensor);  slice_scatter_7 = wait_tensor = None
        copy: "f32[304575880]" = torch.ops.aten.copy.default(slice_scatter_7, wait_tensor);  slice_scatter_7 = wait_tensor = None
        
        # File: <eval_with_key>.96:37 in forward, code: resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(getitem_2, 609102400)
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 609102400)
        
        # File: <eval_with_key>.96:38 in forward, code: view = torch.ops.aten.view.default(copy, [2, -1]);  copy = None
        view: "f32[2, 152287940]" = torch.ops.aten.view.default(copy, [2, -1]);  copy = None
        
        # File: <eval_with_key>.96:39 in forward, code: split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view, [76137800, 6170, 76137800, 6170], 1);  view = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view, [76137800, 6170, 76137800, 6170], 1);  view = None
        getitem_6: "f32[2, 76137800]" = split_with_sizes_1[2];  split_with_sizes_1 = None
        
        # File: <eval_with_key>.96:44 in forward, code: clone_1 = torch.ops.aten.clone.default(getitem_14, memory_format = torch.contiguous_format);  getitem_14 = None
        clone_1: "f32[2, 76137800]" = torch.ops.aten.clone.default(getitem_6, memory_format = torch.contiguous_format);  getitem_6 = None
        
        # File: <eval_with_key>.96:45 in forward, code: view_1 = torch.ops.aten.view.default(clone_1, [152275600]);  clone_1 = None
        view_1: "f32[152275600]" = torch.ops.aten.view.default(clone_1, [152275600]);  clone_1 = None
        
        # File: <eval_with_key>.96:46 in forward, code: as_strided = torch.ops.aten.as_strided.default(view_1, [12340, 12340], [12340, 1], 0);  view_1 = None
        as_strided: "f32[12340, 12340]" = torch.ops.aten.as_strided.default(view_1, [12340, 12340], [12340, 1], 0);  view_1 = None
        
        # File: <eval_with_key>.96:47 in forward, code: copy_ = torch.ops.aten.copy_.default(getitem_2, as_strided);  as_strided = None
        copy_1: "f32[12340, 12340]" = torch.ops.aten.copy.default(arg2_1, as_strided);  as_strided = None
        
        # File: <eval_with_key>.96:48 in forward, code: alias = torch.ops.aten.alias.default(getitem_3)
        alias: "f32[2, 12340]" = torch.ops.aten.alias.default(arg3_1)
        
        # File: <eval_with_key>.96:49 in forward, code: alias_1 = torch.ops.aten.alias.default(alias);  alias = None
        alias_1: "f32[2, 12340]" = torch.ops.aten.alias.default(alias);  alias = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:46 in foreach_all_gather, code: all_gather_output = torch.empty(
        empty_1: "f32[304575880]" = torch.ops.aten.empty.memory_format([304575880], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:49 in foreach_all_gather, code: all_gather_input = all_gather_output.narrow(
        slice_6: "f32[152287940]" = torch.ops.aten.slice.Tensor(empty_1, 0, 0, 152287940)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:52 in foreach_all_gather, code: foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(slice_6, [76137800, 6170, 76137800, 6170]);  slice_6 = None
        getitem_8: "f32[76137800]" = split_with_sizes_2[0]
        getitem_9: "f32[6170]" = split_with_sizes_2[1]
        getitem_10: "f32[76137800]" = split_with_sizes_2[2]
        getitem_11: "f32[6170]" = split_with_sizes_2[3];  split_with_sizes_2 = None
        
        # No stacktrace found for following nodes
        copy__default = torch.ops.aten.copy_.default(getitem_8, arg7_1);  arg7_1 = None
        copy__default_1 = torch.ops.aten.copy_.default(getitem_9, arg8_1);  arg8_1 = None
        copy__default_2 = torch.ops.aten.copy_.default(getitem_10, arg9_1);  arg9_1 = None
        copy__default_3 = torch.ops.aten.copy_.default(getitem_11, arg10_1);  arg10_1 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:54 in foreach_all_gather, code: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
        slice_7: "f32[152287940]" = torch.ops.aten.slice.Tensor(empty_1, 0, 0, 152287940)
        slice_scatter_8: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_7, getitem_8, 0, 0, 76137800);  slice_7 = getitem_8 = None
        slice_scatter_9: "f32[304575880]" = torch.ops.aten.slice_scatter.default(empty_1, slice_scatter_8, 0, 0, 152287940);  empty_1 = slice_scatter_8 = None
        slice_8: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_9, 0, 0, 152287940)
        slice_scatter_10: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_8, getitem_9, 0, 76137800, 76143970);  slice_8 = getitem_9 = None
        slice_scatter_11: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_9, slice_scatter_10, 0, 0, 152287940);  slice_scatter_9 = slice_scatter_10 = None
        slice_9: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 152287940)
        slice_scatter_12: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_9, getitem_10, 0, 76143970, 152281770);  slice_9 = getitem_10 = None
        slice_scatter_13: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, slice_scatter_12, 0, 0, 152287940);  slice_scatter_11 = slice_scatter_12 = None
        slice_10: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 0, 0, 152287940)
        slice_scatter_14: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_10, getitem_11, 0, 152281770, 152287940);  slice_10 = getitem_11 = None
        slice_scatter_15: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_13, slice_scatter_14, 0, 0, 152287940);  slice_scatter_13 = slice_scatter_14 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        slice_15: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_15, 0, 0, 152287940)
        all_gather_into_tensor_1: "f32[304575880]" = torch.ops._c10d_functional.all_gather_into_tensor.default(slice_15, 2, '0');  slice_15 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "f32[304575880]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:977 in all_gather_tensor_inplace, code: return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))
        copy_2: "f32[304575880]" = torch.ops.aten.copy.default(slice_scatter_15, wait_tensor_1);  slice_scatter_15 = wait_tensor_1 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_param.py:455 in unsafe_alloc_storage, code: tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(arg6_1, 609102400)
        resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(arg5_1, 49360)
        resize_storage_bytes__4 = torch.ops.inductor.resize_storage_bytes_.default(copy_1, 609102400)
        resize_storage_bytes__5 = torch.ops.inductor.resize_storage_bytes_.default(arg4_1, 49360)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        view_3: "f32[2, 152287940]" = torch.ops.aten.view.default(copy_2, [2, -1])
        split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(view_3, [76137800, 6170, 76137800, 6170], 1);  view_3 = None
        getitem_36: "f32[2, 76137800]" = split_with_sizes_8[0];  split_with_sizes_8 = None
        clone_2: "f32[2, 76137800]" = torch.ops.aten.clone.default(getitem_36, memory_format = torch.contiguous_format);  getitem_36 = None
        view_4: "f32[152275600]" = torch.ops.aten.view.default(clone_2, [152275600]);  clone_2 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_1: "f32[12340, 12340]" = torch.ops.aten.as_strided.default(view_4, [12340, 12340], [12340, 1], 0);  view_4 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        view_5: "f32[2, 152287940]" = torch.ops.aten.view.default(copy_2, [2, -1])
        split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(view_5, [76137800, 6170, 76137800, 6170], 1);  view_5 = None
        getitem_41: "f32[2, 6170]" = split_with_sizes_9[1];  split_with_sizes_9 = None
        clone_3: "f32[2, 6170]" = torch.ops.aten.clone.default(getitem_41, memory_format = torch.contiguous_format);  getitem_41 = None
        view_6: "f32[12340]" = torch.ops.aten.view.default(clone_3, [12340]);  clone_3 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_2: "f32[12340]" = torch.ops.aten.as_strided.default(view_6, [12340], [1], 0);  view_6 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        view_7: "f32[2, 152287940]" = torch.ops.aten.view.default(copy_2, [2, -1])
        split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(view_7, [76137800, 6170, 76137800, 6170], 1);  view_7 = None
        getitem_46: "f32[2, 76137800]" = split_with_sizes_10[2];  split_with_sizes_10 = None
        clone_4: "f32[2, 76137800]" = torch.ops.aten.clone.default(getitem_46, memory_format = torch.contiguous_format);  getitem_46 = None
        view_8: "f32[152275600]" = torch.ops.aten.view.default(clone_4, [152275600]);  clone_4 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_3: "f32[12340, 12340]" = torch.ops.aten.as_strided.default(view_8, [12340, 12340], [12340, 1], 0);  view_8 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        view_9: "f32[2, 152287940]" = torch.ops.aten.view.default(copy_2, [2, -1]);  copy_2 = None
        split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(view_9, [76137800, 6170, 76137800, 6170], 1);  view_9 = None
        getitem_51: "f32[2, 6170]" = split_with_sizes_11[3];  split_with_sizes_11 = None
        clone_5: "f32[2, 6170]" = torch.ops.aten.clone.default(getitem_51, memory_format = torch.contiguous_format);  getitem_51 = None
        view_10: "f32[12340]" = torch.ops.aten.view.default(clone_5, [12340]);  clone_5 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_4: "f32[12340]" = torch.ops.aten.as_strided.default(view_10, [12340], [1], 0);  view_10 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:137 in foreach_all_gather_copy_out, code: torch._foreach_copy_(out, splits_unpadded)
        _foreach_copy_1 = torch.ops.aten._foreach_copy.default([arg6_1, arg5_1, copy_1, arg4_1], [as_strided_1, as_strided_2, as_strided_3, as_strided_4]);  as_strided_1 = as_strided_2 = as_strided_3 = as_strided_4 = None
        getitem_52: "f32[12340, 12340]" = _foreach_copy_1[0]
        getitem_53: "f32[12340]" = _foreach_copy_1[1]
        getitem_54: "f32[12340, 12340]" = _foreach_copy_1[2]
        getitem_55: "f32[12340]" = _foreach_copy_1[3];  _foreach_copy_1 = None
        
        # File: <eval_with_key>.96:54 in forward, code: mm = torch.ops.aten.mm.default(trace_wrapped, permute_1);  permute_1 = None
        permute_2: "f32[12340, 12340]" = torch.ops.aten.permute.default(getitem_54, [1, 0])
        permute_3: "f32[12340, 12340]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm: "f32[2, 12340]" = torch.ops.aten.mm.default(clone, permute_3);  permute_3 = None
        
        # File: <eval_with_key>.96:55 in forward, code: permute_2 = torch.ops.aten.permute.default(trace_wrapped, [1, 0])
        permute_4: "f32[12340, 2]" = torch.ops.aten.permute.default(clone, [1, 0])
        
        # File: <eval_with_key>.96:56 in forward, code: mm_1 = torch.ops.aten.mm.default(permute_2, getitem_3);  permute_2 = getitem_3 = None
        mm_1: "f32[12340, 12340]" = torch.ops.aten.mm.default(permute_4, arg3_1);  permute_4 = arg3_1 = None
        
        # File: <eval_with_key>.96:57 in forward, code: permute_3 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        permute_5: "f32[12340, 12340]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        
        # File: <eval_with_key>.96:58 in forward, code: sum_1 = torch.ops.aten.sum.dim_IntList(trace_wrapped, [0], True);  trace_wrapped = None
        sum_1: "f32[1, 12340]" = torch.ops.aten.sum.dim_IntList(clone, [0], True);  clone = None
        
        # File: <eval_with_key>.96:59 in forward, code: view_2 = torch.ops.aten.view.default(sum_1, [12340]);  sum_1 = None
        view_11: "f32[12340]" = torch.ops.aten.view.default(sum_1, [12340]);  sum_1 = None
        
        # File: <eval_with_key>.96:60 in forward, code: permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_6: "f32[12340, 12340]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        
        # File: <eval_with_key>.96:61 in forward, code: alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        alias_2: "f32[2, 12340]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
        
        # File: <eval_with_key>.96:62 in forward, code: alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        alias_3: "f32[2, 12340]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        
        # File: <eval_with_key>.96:63 in forward, code: le = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        le: "b8[2, 12340]" = torch.ops.aten.le.Scalar(alias_3, 0);  alias_3 = None
        
        # File: <eval_with_key>.96:64 in forward, code: full = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: <eval_with_key>.96:65 in forward, code: where = torch.ops.aten.where.self(le, full, mm);  le = full = mm = None
        where: "f32[2, 12340]" = torch.ops.aten.where.self(le, full, mm);  le = full = mm = None
        
        # File: <eval_with_key>.96:66 in forward, code: permute_5 = torch.ops.aten.permute.default(where, [1, 0])
        permute_7: "f32[12340, 2]" = torch.ops.aten.permute.default(where, [1, 0])
        
        # File: <eval_with_key>.96:67 in forward, code: mm_2 = torch.ops.aten.mm.default(permute_5, getitem_1);  permute_5 = getitem_1 = None
        mm_2: "f32[12340, 12340]" = torch.ops.aten.mm.default(permute_7, arg1_1);  permute_7 = arg1_1 = None
        
        # File: <eval_with_key>.96:68 in forward, code: permute_6 = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
        permute_8: "f32[12340, 12340]" = torch.ops.aten.permute.default(mm_2, [1, 0]);  mm_2 = None
        
        # File: <eval_with_key>.96:69 in forward, code: sum_2 = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        sum_2: "f32[1, 12340]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        
        # File: <eval_with_key>.96:70 in forward, code: view_3 = torch.ops.aten.view.default(sum_2, [12340]);  sum_2 = None
        view_12: "f32[12340]" = torch.ops.aten.view.default(sum_2, [12340]);  sum_2 = None
        
        # File: <eval_with_key>.96:71 in forward, code: permute_7 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        permute_9: "f32[12340, 12340]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/_dynamo/polyfill.py:43 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        clone_6: "f32[12340]" = torch.ops.aten.clone.default(view_11);  view_11 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/_dynamo/polyfill.py:43 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        clone_7: "f32[12340, 12340]" = torch.ops.aten.clone.default(permute_6);  permute_6 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/_dynamo/polyfill.py:43 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        clone_8: "f32[12340]" = torch.ops.aten.clone.default(view_12);  view_12 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/_dynamo/polyfill.py:43 in accumulate_grad, code: new_grad = torch.clone(new_grad)
        clone_9: "f32[12340, 12340]" = torch.ops.aten.clone.default(permute_9);  permute_9 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:177 in foreach_reduce_scatter, code: reduce_scatter_input = torch.empty(
        empty_2: "f32[304575880]" = torch.ops.aten.empty.memory_format([304575880], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:243 in foreach_reduce_scatter_copy_in, code: grad_views.append(grad.view(world_size, -1))
        view_13: "f32[2, 76137800]" = torch.ops.aten.view.default(clone_9, [2, -1]);  clone_9 = None
        view_14: "f32[2, 6170]" = torch.ops.aten.view.default(clone_8, [2, -1]);  clone_8 = None
        view_15: "f32[2, 76137800]" = torch.ops.aten.view.default(clone_7, [2, -1]);  clone_7 = None
        view_16: "f32[2, 6170]" = torch.ops.aten.view.default(clone_6, [2, -1]);  clone_6 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:249 in foreach_reduce_scatter_copy_in, code: cat_out = torch.cat(grad_views, dim=-1)
        cat: "f32[2, 152287940]" = torch.ops.aten.cat.default([view_13, view_14, view_15, view_16], 1);  view_13 = view_14 = view_15 = view_16 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:250 in foreach_reduce_scatter_copy_in, code: reduce_scatter_input_view = reduce_scatter_input.view(world_size, -1)
        view_17: "f32[2, 152287940]" = torch.ops.aten.view.default(empty_2, [2, -1]);  empty_2 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:252 in foreach_reduce_scatter_copy_in, code: reduce_scatter_input_view.copy_(cat_out)
        copy_3: "f32[2, 152287940]" = torch.ops.aten.copy.default(view_17, cat);  view_17 = cat = None
        view_18: "f32[304575880]" = torch.ops.aten.view.default(copy_3, [304575880]);  copy_3 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:257 in _div_if_needed, code: tensor.div_(div_factor)
        div: "f32[304575880]" = torch.ops.aten.div.Tensor(view_18, 2.0);  view_18 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:189 in foreach_reduce_scatter, code: reduce_scatter_output = reduce_scatter_input.new_empty(
        empty_3: "f32[152287940]" = torch.ops.aten.empty.memory_format([152287940], dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:287 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        reduce_scatter_tensor: "f32[152287940]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(div, 'sum', 2, '0');  div = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "f32[152287940]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:992 in reduce_scatter_tensor_inplace, code: return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))
        copy_4: "f32[152287940]" = torch.ops.aten.copy.default(empty_3, wait_tensor_2);  empty_3 = wait_tensor_2 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:137 in foreach_all_gather_copy_out, code: torch._foreach_copy_(out, splits_unpadded)
        copy_: "f32[12340, 12340]" = torch.ops.aten.copy_.default(arg2_1, getitem_54)
        copy__1: "f32[12340]" = torch.ops.aten.copy_.default(arg4_1, getitem_55)
        copy__2: "f32[12340]" = torch.ops.aten.copy_.default(arg5_1, getitem_53)
        copy__3: "f32[12340, 12340]" = torch.ops.aten.copy_.default(arg6_1, getitem_52)
        
        # No stacktrace found for following nodes
        as_strided_9: "f32[6170, 12340]" = torch.ops.aten.as_strided.default(copy_4, [6170, 12340], [12340, 1], 0)
        view_24: "f32[6170, 12340]" = torch.ops.aten.view.default(as_strided_9, [6170, 12340]);  as_strided_9 = None
        as_strided_10: "f32[6170]" = torch.ops.aten.as_strided.default(copy_4, [6170], [1], 76137800)
        view_25: "f32[6170]" = torch.ops.aten.view.default(as_strided_10, [6170]);  as_strided_10 = None
        as_strided_11: "f32[6170, 12340]" = torch.ops.aten.as_strided.default(copy_4, [6170, 12340], [12340, 1], 76143970)
        view_26: "f32[6170, 12340]" = torch.ops.aten.view.default(as_strided_11, [6170, 12340]);  as_strided_11 = None
        as_strided_12: "f32[6170]" = torch.ops.aten.as_strided.default(copy_4, [6170], [1], 152281770);  copy_4 = None
        view_27: "f32[6170]" = torch.ops.aten.view.default(as_strided_12, [6170]);  as_strided_12 = None
        resize_storage_bytes__default_5 = torch.ops.inductor.resize_storage_bytes_.default(getitem_52, 0);  getitem_52 = None
        resize_storage_bytes__default_6 = torch.ops.inductor.resize_storage_bytes_.default(arg6_1, 0);  arg6_1 = None
        resize_storage_bytes__default_7 = torch.ops.inductor.resize_storage_bytes_.default(copy_1, 0)
        resize_storage_bytes__default_8 = torch.ops.inductor.resize_storage_bytes_.default(getitem_53, 0);  getitem_53 = None
        resize_storage_bytes__default_9 = torch.ops.inductor.resize_storage_bytes_.default(arg4_1, 0);  arg4_1 = None
        resize_storage_bytes__default_10 = torch.ops.inductor.resize_storage_bytes_.default(arg5_1, 0);  arg5_1 = None
        resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(getitem_54, 0)
        resize_storage_bytes__default_12 = torch.ops.inductor.resize_storage_bytes_.default(arg2_1, 0);  arg2_1 = None
        resize_storage_bytes__default_13 = torch.ops.inductor.resize_storage_bytes_.default(copy_1, 0);  copy_1 = None
        resize_storage_bytes__default_14 = torch.ops.inductor.resize_storage_bytes_.default(getitem_54, 0);  getitem_54 = None
        resize_storage_bytes__default_15 = torch.ops.inductor.resize_storage_bytes_.default(getitem_55, 0);  getitem_55 = None
        return [view_24, view_25, view_26, view_27]