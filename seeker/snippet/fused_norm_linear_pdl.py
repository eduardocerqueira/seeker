#date: 2026-01-16T17:20:21Z
#url: https://api.github.com/gists/b0d32ebf4d383746fceed6d67a215254
#owner: https://api.github.com/users/mattteochen

"""
Standalone reproduction of gpt_oss norm + router pattern for PDL profiling.
Mimics exactly what happens in gpt_oss.py with our modifications:
  - layer_communicator.prepare_mlp() → fused_add_rmsnorm
  - self.router() → F.linear (small GEMM)

Run with nsys:
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -o gpt_oss_norm_router \
        python python/dev/repro_gpt_oss_norm_router.py
"""

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

from sgl_kernel import fused_add_rmsnorm

USE_PDL = True

def main():
    device = "cuda"
    dtype = torch.bfloat16
    
    # ========== GptOss model config ==========
    # These match the gpt-oss-20b model
    num_tokens = 1  # decode phase: "**********"
    hidden_size = 2880
    num_experts = 32
    num_layers = 40
    eps = 1e-6  # rms_norm_eps
    
    warmup_iters = 50
    profile_iters = 10
    
    print("=" * 60)
    print("GptOss Norm + Router PDL Profiling (Standalone)")
    print("=" * 60)
    print(f"  num_tokens: "**********"
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_experts: {num_experts}")
    print(f"  num_layers: {num_layers}")
    print(f"  eps: {eps}")
    print(f"  dtype: {dtype}")
    print(f"  warmup_iters: {warmup_iters}")
    print(f"  profile_iters: {profile_iters}")
    print("=" * 60)
    
    # ========== Create per-layer weights (matching gpt_oss) ==========
    layer_weights = []
    for i in range(num_layers):
        layer = {
            # post_attention_layernorm weight: [hidden_size]
            'post_attn_norm': torch.randn(hidden_size, dtype=dtype, device=device).contiguous(),
            # router (ReplicatedLinear): weight [num_experts, hidden_size], bias [num_experts]
            'router_weight': torch.randn(num_experts, hidden_size, dtype=dtype, device=device).contiguous(),
            'router_bias': torch.randn(num_experts, dtype=dtype, device=device).contiguous(),
        }
        layer_weights.append(layer)
    
    # ========== Static tensors for CUDA graph capture ==========
    # These match the shapes in gpt_oss decode phase
    static_hidden_states = "**********"=dtype, device=device).contiguous()
    static_residual = "**********"=dtype, device=device).contiguous()
    
    print("\nTensor shapes:")
    print(f"  hidden_states: "**********"
    print(f"  residual: "**********"
    print(f"  post_attn_norm weight: [{hidden_size}]")
    print(f"  router_weight: [{num_experts}, {hidden_size}]")
    print(f"  router_bias: [{num_experts}]")
    print(f"\nRouter GEMM: "**********"
    
    # ========== Workload function (matching gpt_oss forward) ==========
    def workload():
        """
        Mimics GptOssDecoderLayer.forward() with our modifications:
        - prepare_mlp() calls fused_add_rmsnorm
        - mlp.forward_normal() only calls router
        """
        hidden_states = static_hidden_states
        residual = static_residual
        
        for layer_idx in range(num_layers):
            layer = layer_weights[layer_idx]
            
            # === layer_communicator.prepare_mlp() ===
            # This calls post_attention_layernorm(hidden_states, residual)
            # Which internally calls fused_add_rmsnorm
            # Signature: fused_add_rmsnorm(x, residual, weight, eps, enable_pdl)
            fused_add_rmsnorm(
                hidden_states,
                residual, 
                layer['post_attn_norm'],
                eps,
                USE_PDL,
            )
            # After fused_add_rmsnorm:
            # - hidden_states is normalized (in-place)
            # - residual = hidden_states + residual (in-place)
            
            # === mlp.forward_normal() - only router ===
            # router is ReplicatedLinear which uses F.linear
            router_logits = F.linear(hidden_states, layer['router_weight'], layer['router_bias'])
            
            # In full model, topk + experts would run here, but we skip them
            # hidden_states stays the same for next layer (since we skip MLP)
        
        return hidden_states
    
    # ========== Warmup ==========
    print("\nWarming up...")
    for _ in range(warmup_iters):
        workload()
    torch.cuda.synchronize()
    print("Warmup done.")
    
    # ========== Capture CUDA Graph ==========
    print("\nCapturing CUDA graph...")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Warmup in capture stream
        for _ in range(3):
            workload()
        
        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            workload()
    torch.cuda.synchronize()
    print("  Graph captured.")
    
    # ========== Profile ==========
    print("\n" + "=" * 60)
    print("Profiling with CUDA Graph")
    print("=" * 60)
    
    torch.cuda.cudart().cudaProfilerStart()
    
    nvtx.range_push("gpt_oss_norm_router")
    for i in range(profile_iters):
        nvtx.range_push(f"iter_{i}")
        graph.replay()
        nvtx.range_pop()
    torch.cuda.synchronize()
    nvtx.range_pop()
    
    torch.cuda.cudart().cudaProfilerStop()
    
    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    print("\nTo analyze, run:")
    print("  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -o gpt_oss_norm_router \\")
    print("      python python/dev/repro_gpt_oss_norm_router.py")
    print("  nsys-ui gpt_oss_norm_router.nsys-rep")
    print("\nExpected kernels per layer:")
    print("  - FusedAddRMSNormKernel (fused_add_rmsnorm)")
    print("  - cuBLAS GEMM (router linear)")
    print("\nWith PDL enabled, the norm kernel should overlap with the next GEMM.")


if __name__ == "__main__":
    main()

 (fused_add_rmsnorm)")
    print("  - cuBLAS GEMM (router linear)")
    print("\nWith PDL enabled, the norm kernel should overlap with the next GEMM.")


if __name__ == "__main__":
    main()

