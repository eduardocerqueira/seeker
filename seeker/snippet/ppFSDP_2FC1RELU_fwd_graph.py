#date: 2024-03-07T18:37:10Z
#url: https://api.github.com/gists/ae8e7022bb4605d206d7839cad2ee5b8
#owner: https://api.github.com/users/yf225

TRACED GRAPH
 ===== Forward graph 0 =====
 /data/users/willfeng/pytorch_yf225/torch/fx/_lazy_graph_module.py class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 12340]", primals_2: "f32[76137800]", primals_3: "f32[6170]", primals_4: "f32[76137800]", primals_5: "f32[6170]", primals_6: "f32[12340, 12340]", primals_7: "f32[12340]", primals_8: "f32[12340, 12340]", primals_9: "f32[12340]", primals_10):
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:46 in foreach_all_gather, code: all_gather_output = torch.empty(
        empty: "f32[304575880]" = torch.ops.aten.empty.memory_format([304575880], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:49 in foreach_all_gather, code: all_gather_input = all_gather_output.narrow(
        slice_1: "f32[152287940]" = torch.ops.aten.slice.Tensor(empty, 0, 0, 152287940)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:52 in foreach_all_gather, code: foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(slice_1, [76137800, 6170, 76137800, 6170])
        getitem: "f32[76137800]" = split_with_sizes[0]
        getitem_1: "f32[6170]" = split_with_sizes[1]
        getitem_2: "f32[76137800]" = split_with_sizes[2]
        getitem_3: "f32[6170]" = split_with_sizes[3];  split_with_sizes = None
        
        # No stacktrace found for following nodes
        copy__default = torch.ops.aten.copy_.default(getitem, primals_2);  primals_2 = None
        copy__default_1 = torch.ops.aten.copy_.default(getitem_1, primals_3);  primals_3 = None
        copy__default_2 = torch.ops.aten.copy_.default(getitem_2, primals_4);  primals_4 = None
        copy__default_3 = torch.ops.aten.copy_.default(getitem_3, primals_5);  primals_5 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:54 in foreach_all_gather, code: torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
        slice_scatter: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_1, getitem, 0, 0, 76137800);  slice_1 = getitem = None
        slice_scatter_1: "f32[304575880]" = torch.ops.aten.slice_scatter.default(empty, slice_scatter, 0, 0, 152287940);  empty = slice_scatter = None
        slice_3: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 152287940)
        slice_scatter_2: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_3, getitem_1, 0, 76137800, 76143970);  slice_3 = getitem_1 = None
        slice_scatter_3: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, slice_scatter_2, 0, 0, 152287940);  slice_scatter_1 = slice_scatter_2 = None
        slice_4: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 0, 0, 152287940)
        slice_scatter_4: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_4, getitem_2, 0, 76143970, 152281770);  slice_4 = getitem_2 = None
        slice_scatter_5: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, slice_scatter_4, 0, 0, 152287940);  slice_scatter_3 = slice_scatter_4 = None
        slice_5: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 152287940)
        slice_scatter_6: "f32[152287940]" = torch.ops.aten.slice_scatter.default(slice_5, getitem_3, 0, 152281770, 152287940);  slice_5 = getitem_3 = None
        slice_scatter_7: "f32[304575880]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_6, 0, 0, 152287940);  slice_scatter_5 = slice_scatter_6 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:229 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        slice_10: "f32[152287940]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 0, 152287940)
        all_gather_into_tensor: "f32[304575880]" = torch.ops._c10d_functional.all_gather_into_tensor.default(slice_10, 2, '0');  slice_10 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:144 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "f32[304575880]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_functional_collectives.py:977 in all_gather_tensor_inplace, code: return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))
        copy: "f32[304575880]" = torch.ops.aten.copy.default(slice_scatter_7, wait_tensor);  slice_scatter_7 = wait_tensor = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_param.py:455 in unsafe_alloc_storage, code: tensor.untyped_storage().resize_(tensor.numel() * tensor.itemsize)
        resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 609102400)
        resize_storage_bytes__1 = torch.ops.inductor.resize_storage_bytes_.default(primals_7, 49360)
        resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(primals_8, 609102400)
        resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(primals_9, 49360)
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        view_1: "f32[2, 152287940]" = torch.ops.aten.view.default(copy, [2, -1]);  copy = None
        split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_1, [76137800, 6170, 76137800, 6170], 1);  view_1 = None
        getitem_28: "f32[2, 76137800]" = split_with_sizes_6[0]
        clone: "f32[2, 76137800]" = torch.ops.aten.clone.default(getitem_28, memory_format = torch.contiguous_format);  getitem_28 = None
        view_2: "f32[152275600]" = torch.ops.aten.view.default(clone, [152275600]);  clone = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided: "f32[12340, 12340]" = torch.ops.aten.as_strided.default(view_2, [12340, 12340], [12340, 1], 0);  view_2 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        getitem_33: "f32[2, 6170]" = split_with_sizes_6[1]
        clone_1: "f32[2, 6170]" = torch.ops.aten.clone.default(getitem_33, memory_format = torch.contiguous_format);  getitem_33 = None
        view_4: "f32[12340]" = torch.ops.aten.view.default(clone_1, [12340]);  clone_1 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_1: "f32[12340]" = torch.ops.aten.as_strided.default(view_4, [12340], [1], 0);  view_4 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        getitem_38: "f32[2, 76137800]" = split_with_sizes_6[2]
        clone_2: "f32[2, 76137800]" = torch.ops.aten.clone.default(getitem_38, memory_format = torch.contiguous_format);  getitem_38 = None
        view_6: "f32[152275600]" = torch.ops.aten.view.default(clone_2, [152275600]);  clone_2 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_2: "f32[12340, 12340]" = torch.ops.aten.as_strided.default(view_6, [12340, 12340], [12340, 1], 0);  view_6 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:127 in foreach_all_gather_copy_out, code: splits[i].contiguous().view(splits[i].numel()),
        getitem_43: "f32[2, 6170]" = split_with_sizes_6[3];  split_with_sizes_6 = None
        clone_3: "f32[2, 6170]" = torch.ops.aten.clone.default(getitem_43, memory_format = torch.contiguous_format);  getitem_43 = None
        view_8: "f32[12340]" = torch.ops.aten.view.default(clone_3, [12340]);  clone_3 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/distributed/_composable/fsdp/_fsdp_collectives.py:126 in foreach_all_gather_copy_out, code: torch.as_strided(
        as_strided_3: "f32[12340]" = torch.ops.aten.as_strided.default(view_8, [12340], [1], 0);  view_8 = None
        
        # No stacktrace found for following nodes
        copy__default_4 = torch.ops.aten.copy_.default(primals_6, as_strided);  as_strided = None
        copy__default_5 = torch.ops.aten.copy_.default(primals_7, as_strided_1);  as_strided_1 = None
        copy__default_6 = torch.ops.aten.copy_.default(primals_8, as_strided_2);  as_strided_2 = None
        copy__default_7 = torch.ops.aten.copy_.default(primals_9, as_strided_3);  as_strided_3 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_1: "f32[12340, 12340]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm: "f32[2, 12340]" = torch.ops.aten.addmm.default(primals_7, primals_1, permute_1);  permute_1 = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/nn/modules/activation.py:103 in forward, code: return F.relu(input, inplace=self.inplace)
        relu: "f32[2, 12340]" = torch.ops.aten.relu.default(addmm);  addmm = None
        
        # File: /data/users/willfeng/pytorch_yf225/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_3: "f32[12340, 12340]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_1: "f32[2, 12340]" = torch.ops.aten.addmm.default(primals_9, relu, permute_3);  permute_3 = None
        
        # No stacktrace found for following nodes
        resize_storage_bytes__default_4 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0)
        resize_storage_bytes__default_5 = torch.ops.inductor.resize_storage_bytes_.default(primals_6, 0);  primals_6 = None
        resize_storage_bytes__default_6 = torch.ops.inductor.resize_storage_bytes_.default(primals_9, 0)
        resize_storage_bytes__default_7 = torch.ops.inductor.resize_storage_bytes_.default(primals_9, 0);  primals_9 = None
        resize_storage_bytes__default_8 = torch.ops.inductor.resize_storage_bytes_.default(primals_7, 0)
        resize_storage_bytes__default_9 = torch.ops.inductor.resize_storage_bytes_.default(primals_8, 0)
        resize_storage_bytes__default_10 = torch.ops.inductor.resize_storage_bytes_.default(primals_7, 0);  primals_7 = None
        resize_storage_bytes__default_11 = torch.ops.inductor.resize_storage_bytes_.default(primals_8, 0)
        return [addmm_1, primals_1, primals_8, relu]