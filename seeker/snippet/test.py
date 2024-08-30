#date: 2024-08-30T17:10:20Z
#url: https://api.github.com/gists/215f0c315c532c90b8e7d1310596834a
#owner: https://api.github.com/users/youkaichao

import torch
from typing import Optional

from torch._dynamo.backends.common import aot_autograd

@torch.library.custom_op("custom::paged_attention", mutates_args=[])
def paged_attention(x: "**********": torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    num_prefill_tokens = "**********"
    bs = x.size(0)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"p "**********"r "**********"e "**********"f "**********"i "**********"l "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"= "**********"= "**********"  "**********"0 "**********": "**********"
        ... # call decode attention
    else:
        ... # call prefill attention with x[: "**********"
        ... # call decode attention with x[num_prefill_tokens: "**********"
    return output

@paged_attention.register_fake
def _(x: "**********": torch.Tensor, cache: torch.Tensor):
    return torch.empty_like(x)

def attention(x: "**********": torch.Tensor, cache: Optional[torch.Tensor] = None):
    if cache is not None:
        return torch.ops.custom.paged_attention(x, num_prefill_tokens, cache)
    return x * 2

eager_model = True

def custom_compiler(gm, inputs):

    # compilation options
    # option 1: pass the full graph to inductor
    # option 2: run the model in eager mode
    # option 3: find subgraph and replace with kernels inside vLLM

    print(gm._graph.python_code(root_module="self", verbose=True).src)

    # selction logic
    static_shape_graphs = dict()
    dynamic_shape_graph = None
    def forward(*args, **kwargs):
        nonlocal static_shape_graphs, dynamic_shape_graph
        batchsize = ... # Question: how to get batchsize from args?
        if dynamic_shape_graph is None:
            # if the input is symbolic shape, compile with dynamic shape support
            dynamic_shape_graph = gm.forward

        if eager_model:
            return dynamic_shape_graph(*args, **kwargs)

        if batchsize not in static_shape_graphs:
            # if the input is static shape, compile with static shape support
            static_shape_graphs[batchsize] = gm.forward
        return static_shape_graphs[batchsize](*args, **kwargs)

    return forward

def target_fn(x, num_prefill_tokens: "**********":
    x = (x + 1) * 5
    if cache is not None:
        x = "**********"
    else:
        x = x * 2
    x = x.sin()
    x = x.cos()
    return x

compiled_target_fn = torch.compile(backend=aot_autograd(fw_compiler=custom_compiler))(target_fn)

compiled_codes = []

def hook(old_colde, new_code):
    if old_colde is target_fn.__code__:
        compiled_codes.append(new_code)

torch._dynamo.convert_frame.register_bytecode_hook(hook)

def dispatcher(x, num_prefill_tokens: "**********":
    if len(compiled_codes) < 2:
        return compiled_target_fn(x, num_prefill_tokens, cache)
    else:
        target_fn.__code__ = compiled_codes[1]
        return target_fn(x, num_prefill_tokens, cache)

def test():

    # profile run, without kv cache, fully static shape, max size
    num_prefill_tokens = "**********"=torch.int32)
    dispatcher(torch.randn(20, 10), num_prefill_tokens, None)

    # create cache
    cache = torch.randn(1, 10)

    # warmup run, mark the input tensor as dynamic
    x = torch.randn(10, 10)
    torch._dynamo.mark_dynamic(x, 0)
    num_prefill_tokens = "**********"=torch.int32)
    out = "**********"
    print(out)

    # the following run with not trigger Dynamo/Aot Autograd

    # if we are using `--enforce-eager`, we want this to directly run
    # with compiled kernel that can handle dynamic shape
    y = torch.randn(5, 10)
    num_prefill_tokens = "**********"=torch.int32)
    out = "**********"
    print(out)

    eager_model = False

    # if we are using cudagraph, this is an additional warmup to capture cuda graph
    for i in [1, 2, 4, 8, 16]:
        y = torch.randn(i, 10)
        num_prefill_tokens = "**********"=torch.int32)
        out = "**********"
    # and then, for later runs, we can directly run with compiled kernel if the shape
    # matches the recorded shape. if not, run with dynamic shape
    y = torch.randn(4, 10)
    num_prefill_tokens = "**********"=torch.int32)
    out = "**********"
    print(out)

if __name__ == "__main__":
    test()
