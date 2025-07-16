#date: 2025-07-16T17:09:41Z
#url: https://api.github.com/gists/2bd097bee77538ce957fcea7881da45e
#owner: https://api.github.com/users/fabianmcg

import numpy as np
import torch
import torch.nn as nn
import iree.turbine.aot as aot

from iree.compiler import compile_file, OutputFormat
from iree.runtime import load_vm_flatbuffer, DeviceArray

from tempfile import TemporaryDirectory
from torch.autograd import DeviceType
from torch.profiler import profile as torch_profile, ProfilerActivity
from pathlib import Path

from typing import Sequence, Any


def _compile_exported(exported: aot.ExportOutput, **kwargs):
    buffer = None
    with TemporaryDirectory() as tmp:
        exported_name = Path(tmp) / "exported.mlirbc"
        exported.save_mlir(str(exported_name))
        buffer = compile_file(str(exported_name), **kwargs)

    assert buffer is not None
    return buffer


def _from_torch(v: Any):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return v


def _from_iree(v: Any):
    if isinstance(v, DeviceArray):
        return v.to_host()
    return v


def rel_error(x_true: np.ndarray, x: np.ndarray):
    x_true = x_true.astype(np.float64)
    x = x.astype(np.float64)
    return np.linalg.norm(x - x_true) / np.linalg.norm(x)


class Run:
    def __init__(self, arguments: Sequence[Any]):
        self.arguments = list(arguments)

    def run(self):
        pass

    def profile(
        self, num_its: int = 10, print_profile: bool = True, row_limit: int = 20
    ):
        with torch_profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=num_its),
            record_shapes=True,
        ) as prof:
            for i in range(num_its + 6):
                self.run()
                prof.step()
        events = prof.key_averages()
        if print_profile:
            print(events.table(sort_by="self_cuda_time_total", row_limit=row_limit))
        return np.array(
            [
                event.self_device_time_total
                for event in events
                if event.device_type == DeviceType.CUDA
            ]
        ).sum() / (num_its * 1000.0)


class TorchModule(Run):
    def __init__(self, module: nn.Module, arguments: Sequence[Any]):
        super().__init__(arguments)
        self.module = module

    def run(self):
        return self.module(*self.arguments)


class IREEModule(Run):
    def __init__(self, vmfb_bytes: bytes, arguments: Sequence[Any]):
        super().__init__(arguments)
        self.vmfb_bytes = vmfb_bytes
        self.vmfb = load_vm_flatbuffer(self.vmfb_bytes, driver="hip")

    @staticmethod
    def from_torch(module: nn.Module, arguments: Sequence[Any], **kwargs):
        exported = aot.export(
            module,
            args=(*arguments,),
            import_symbolic_shape_expressions=True,
            **kwargs,
        )
        vmfb_bytes = _compile_exported(
            exported,
            target_backends=["rocm"],
            optimize=True,
            extra_args=[
                "--iree-hip-target=gfx942",
                "--iree-opt-level=O3",
                "--iree-opt-strip-assertions=true",
            ],
            output_format=OutputFormat.FLATBUFFER_BINARY,
            strip_source_map=True,
            strip_debug_ops=True,
            output_mlir_debuginfo=False,
        )
        return IREEModule(vmfb_bytes, [_from_torch(a) for a in arguments])

    def run(self):
        return self.vmfb["main"](*self.arguments)


def main():
    class CausalAttention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

    model = CausalAttention()
    bs = 4
    num_heads = 16
    seq_len = 1011
    head_dim = 64

    q = torch.randn((bs, num_heads, 1, head_dim)).cuda()
    k = torch.randn((bs, num_heads, seq_len, head_dim)).cuda()
    v = torch.randn((bs, num_heads, seq_len, head_dim)).cuda()
    dyn_seq_len = torch.export.Dim("seq_len")

    iree_run = IREEModule.from_torch(
        model,
        (q, k, v),
        dynamic_shapes={
            "q": {},  # causal or prefill static
            "k": {2: dyn_seq_len},
            "v": {2: dyn_seq_len},
        },
    )

    torch_run = TorchModule(model, (q, k, v))

    print(f"Total torch time: {torch_run.profile():.3f} us")
    print(f"Total IREE time: {iree_run.profile():.3f} us")
    print(
        f"Numeric error: {rel_error(_from_torch(torch_run.run()), _from_iree(iree_run.run()))}"
    )


if __name__ == "__main__":
    main()
