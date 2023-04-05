#date: 2023-04-05T16:57:42Z
#url: https://api.github.com/gists/2dc45a6ac3c54a0ecfc48b3cfc2a6d68
#owner: https://api.github.com/users/pashu123


import torch
from diffusers import StableDiffusionPipeline
import torch_mlir
from shark.shark_importer import import_with_fx
import os
import torch.fx as fx
import sys

model_input = {
    "clip": (torch.randint(1, 2, (1, 77)),),
    "vae": (torch.randn(1, 4, 128, 128),),
    "unet": (
        torch.randn(2, 4, 96, 96).cuda(),  # latents
        torch.tensor([1]).float().cuda(),  # timestep
        torch.randn(2, 77, 1024).cuda(),  # embedding
    ),
}

def compile_via_shark(model, inputs):
    # import torch_mlir
    # import io
    # bytecode_stream = io.BytesIO()
    # import sys
    # linalg_ir = torch_mlir.compile(model, inputs, output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
    # linalg_ir.operation.write_bytecode(bytecode_stream)
    # bytecode = bytecode_stream.getvalue()

    model = model.float()
    is_f16 = True 
    input_mask = [True, True, True]
    bytecode = import_with_fx(model, inputs, is_f16=is_f16, f16_input_mask=input_mask)
    print(bytecode.graph)
    inputs = [x.half() for x in inputs]
    print(bytecode(*inputs))
    return bytecode
    with open(os.path.join("xyz.mlir"), "wb") as mlir_file:
        mlir_file.write(bytecode[0])

    sys.exit()
    # fx_g = fx.symbolic_trace(model)
    # print(fx_g.graph)

    # bytecode = import_with_fx(model, inputs)
    # return bytecode

    from shark.shark_inference import SharkInference
    shark_module = SharkInference(
        mlir_module=bytecode[0], device="vulkan", mlir_dialect="tm_tensor",
    )
    # extra_args = ['--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-detach-elementwise-from-named-ops,iree-flow-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))', '--iree-spirv-index-bits=64']
    shark_module.compile(extra_args=[])
    return shark_module

class UNetWrapper(torch.nn.Module):
    
    def __init__(self, shark_unet):
        super().__init__()
        self.wrapped_unet = shark_unet
        self.in_channels = None
        self.device = None
        self.config = None
    
    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        # sample_np = sample.detach().cpu().numpy()
        # timestep_np = timestep.half().detach().cpu().reshape(-1).numpy()
        # encoder_hidden_states_np = encoder_hidden_states.detach().cpu().numpy()
        # inputs = [sample_np, timestep_np, encoder_hidden_states_np]
        sample = self.wrapped_unet(sample, timestep, encoder_hidden_states)
        # rest of the pipeline is always in float16
        return sample

class UnetCustom(torch.nn.Module):
    def __init__(self, pipe_unet):
        super().__init__()
        self.unet = pipe_unet
        self.in_channels = None
        self.device = None
        self.config = None

    def forward(self, latent, timestep, text_embedding):
        unet_out = self.unet.forward(
            latent,
            timestep,
            text_embedding,
            return_dict=False,
        )[0]
        return unet_out

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
# pipe.enable_attention_slicing()

unet_graph = UnetCustom(pipe.unet)

unet_graph.in_channels = pipe.unet.in_channels
unet_graph.device = pipe.unet.device
unet_graph.config = pipe.unet.config

del pipe.unet
pipe.unet = unet_graph

shark_unet = compile_via_shark(pipe.unet, model_input["unet"])
# shark_unet = shark_unet.cuda()
unet_graph = UNetWrapper(shark_unet)

unet_graph.in_channels = pipe.unet.in_channels
unet_graph.device = pipe.unet.device
unet_graph.config = pipe.unet.config

del pipe.unet
pipe.unet = unet_graph


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save(f"astronaut_rides_horse.png")


# import torch
# from shark.shark_inference import SharkInference
# from shark.shark_importer import import_with_fx
# from typing import List

# import torch_mlir
# from torch_mlir.dynamo import make_simple_dynamo_backend
# import torch._dynamo as dynamo
# from torch.fx.experimental.proxy_tensor import make_fx
# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
# from shark.shark_inference import SharkInference
# from io import BytesIO

# def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    # removed_indexes = []
    # for node in fx_g.graph.nodes:
        # if node.op == "output":
            # assert (
                # len(node.args) == 1
            # ), "Output node must have a single argument"
            # node_arg = node.args[0]
            # if isinstance(node_arg, (list, tuple)):
                # node_arg = list(node_arg)
                # node_args_len = len(node_arg)
                # for i in range(node_args_len):
                    # curr_index = node_args_len - (i + 1)
                    # if node_arg[curr_index] is None:
                        # removed_indexes.append(curr_index)
                        # node_arg.pop(curr_index)
                # node.args = (tuple(node_arg),)
                # break

    # if len(removed_indexes) > 0:
        # fx_g.graph.lint()
        # fx_g.graph.eliminate_dead_code()
        # fx_g.recompile()
    # removed_indexes.sort()
    # return removed_indexes


# def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    # """
    # Replace tuple with tuple element in functions that return one-element tuples.
    # Returns true if an unwrapping took place, and false otherwise.
    # """
    # unwrapped_tuple = False
    # for node in fx_g.graph.nodes:
        # if node.op == "output":
            # assert (
                # len(node.args) == 1
            # ), "Output node must have a single argument"
            # node_arg = node.args[0]
            # if isinstance(node_arg, tuple):
                # if len(node_arg) == 1:
                    # node.args = (node_arg[0],)
                    # unwrapped_tuple = True
                    # break

    # if unwrapped_tuple:
        # fx_g.graph.lint()
        # fx_g.recompile()
    # return unwrapped_tuple


# def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    # for node in fx_g.graph.nodes:
        # if node.op == "output":
            # assert (
                # len(node.args) == 1
            # ), "Output node must have a single argument"
            # node_arg = node.args[0]
            # if isinstance(node_arg, tuple):
                # return len(node_arg) == 0
    # return False


# def transform_fx(fx_g):
    # for node in fx_g.graph.nodes:
        # if node.op == "call_function":
            # if node.target in [
                # torch.ops.aten.empty,
            # ]:
                # # aten.empty should be filled with zeros.
                # if node.target in [torch.ops.aten.empty]:
                    # with fx_g.graph.inserting_after(node):
                        # new_node = fx_g.graph.call_function(
                            # torch.ops.aten.zero_,
                            # args=(node,),
                        # )
                        # node.append(new_node)
                        # node.replace_all_uses_with(new_node)
                        # new_node.args = (node,)

    # fx_g.graph.lint()


# @make_simple_dynamo_backend
# def refbackend_torchdynamo_backend(
    # fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
# ):
    # # handling usage of empty tensor without initializing
    # transform_fx(fx_graph)
    # fx_graph.recompile()
    # if _returns_nothing(fx_graph):
        # return fx_graph
    # removed_none_indexes = _remove_nones(fx_graph)
    # was_unwrapped = _unwrap_single_tuple_return(fx_graph)

    # mlir_module = torch_mlir.compile(
        # fx_graph, example_inputs, output_type="linalg-on-tensors"
    # )
    # mlir_module.dump()

    # bytecode_stream = BytesIO()
    # mlir_module.operation.write_bytecode(bytecode_stream)
    # bytecode = bytecode_stream.getvalue()

    # shark_module = SharkInference(
        # mlir_module=bytecode, device="vulkan", mlir_dialect="tm_tensor"
    # )
    # shark_module.compile()

    # def compiled_callable(*inputs):
        # inputs = [x.numpy() for x in inputs]
        # result = shark_module("forward", inputs)
        # if was_unwrapped:
            # result = [
                # result,
            # ]
        # if not isinstance(result, list):
            # result = torch.from_numpy(result)
        # else:
            # result = tuple(torch.from_numpy(x) for x in result)
            # result = list(result)
            # for removed_index in removed_none_indexes:
                # result.insert(removed_index, None)
            # result = tuple(result)
        # return result

    # return compiled_callable

# torch._dynamo.config.suppress_errors = True

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = "**********"
# model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").eval()


# @torch.inference_mode()
# def shark_module(model, input):
    # return model(input)


# sequence = "Hey I am doing just right"

# tokenized_inputs = "**********"="pt")
# print(tokenized_inputs["input_ids"].shape)
# print(tokenized_inputs["input_ids"].dtype)

# # model(tokenized_inputs["input_ids"])
# inputs = "**********"


# dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(shark_module)

# x = dynamo_callable(model, inputs)
# print(x)

# def compile_via_shark(model, inputs):
    # with torch.no_grad():
        # return model(*inputs)

# dynamo_callable = dynamo.optimize(refbackend_torchdynamo_backend)(compile_via_shark)


# model_input = {
    # "clip": (torch.randint(1, 2, (1, 77)),),
    # "vae": (torch.randn(1, 4, 128, 128),),
    # "unet": (
        # torch.randn(2, 4, 96, 96),  # latents
        # torch.tensor([1]).float(),  # timestep
        # torch.randn(2, 77, 1024),  # embedding
    # ),
# }

# x = dynamo_callable(pipe.unet.eval(), model_input["unet"])
  # embedding
    # ),
# }

# x = dynamo_callable(pipe.unet.eval(), model_input["unet"])
