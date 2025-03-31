#date: 2025-03-31T17:06:57Z
#url: https://api.github.com/gists/1cc381306be45f05b754e1767f9c8d58
#owner: https://api.github.com/users/kasukanra

import argparse
import os
import json
from pathlib import Path
import safetensors.torch
import torch
import tqdm
from collections import OrderedDict

"""
Extract Flux/Flex Transformer from All-in-One Model

This script extracts the transformer part from a Flux/Flex all-in-one model.
It reverses the concatenation process that was used to create the all-in-one model,
and saves the extracted model in the standard diffusers format with multiple safetensors files.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the all-in-one model file (.safetensors)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the extracted transformer model",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        help="Path to a reference transformer model for file structure",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of double transformer blocks",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=38,
        help="Number of single transformer blocks",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=24,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=128,
        help="Dimension of each attention head",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose information about tensor shapes",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load the all-in-one model
    print(f"Loading model from {args.input_path}")
    state_dict = safetensors.torch.load_file(args.input_path)
    
    # Define the transformer prefix in the all-in-one model
    transformer_prefix = "model.diffusion_model."
    
    # Create transformer state dict
    print("Extracting transformer tensors")
    transformer_state_dict = OrderedDict()
    
    # Process time_in, vector_in, guidance_in, txt_in, img_in
    input_mapping = {
        # time_in
        f"{transformer_prefix}time_in.in_layer.weight": "time_text_embed.timestep_embedder.linear_1.weight",
        f"{transformer_prefix}time_in.in_layer.bias": "time_text_embed.timestep_embedder.linear_1.bias",
        f"{transformer_prefix}time_in.out_layer.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        f"{transformer_prefix}time_in.out_layer.bias": "time_text_embed.timestep_embedder.linear_2.bias",
        
        # vector_in
        f"{transformer_prefix}vector_in.in_layer.weight": "time_text_embed.text_embedder.linear_1.weight",
        f"{transformer_prefix}vector_in.in_layer.bias": "time_text_embed.text_embedder.linear_1.bias",
        f"{transformer_prefix}vector_in.out_layer.weight": "time_text_embed.text_embedder.linear_2.weight",
        f"{transformer_prefix}vector_in.out_layer.bias": "time_text_embed.text_embedder.linear_2.bias",
        
        # guidance_in
        f"{transformer_prefix}guidance_in.in_layer.weight": "time_text_embed.guidance_embedder.linear_1.weight",
        f"{transformer_prefix}guidance_in.in_layer.bias": "time_text_embed.guidance_embedder.linear_1.bias",
        f"{transformer_prefix}guidance_in.out_layer.weight": "time_text_embed.guidance_embedder.linear_2.weight",
        f"{transformer_prefix}guidance_in.out_layer.bias": "time_text_embed.guidance_embedder.linear_2.bias",
        
        # txt_in, img_in
        f"{transformer_prefix}txt_in.weight": "context_embedder.weight",
        f"{transformer_prefix}txt_in.bias": "context_embedder.bias",
        f"{transformer_prefix}img_in.weight": "x_embedder.weight",
        f"{transformer_prefix}img_in.bias": "x_embedder.bias",
        
        # final_layer
        f"{transformer_prefix}final_layer.linear.weight": "proj_out.weight",
        f"{transformer_prefix}final_layer.linear.bias": "proj_out.bias",
    }
    
    for all_in_one_key, diffusers_key in input_mapping.items():
        if all_in_one_key in state_dict:
            transformer_state_dict[diffusers_key] = state_dict[all_in_one_key]
            if args.verbose:
                print(f"Extracted: {all_in_one_key} -> {diffusers_key}")
    
    # Handle adaLN modulation special case - swapping scale and shift
    adaLN_weight_key = f"{transformer_prefix}final_layer.adaLN_modulation.1.weight"
    adaLN_bias_key = f"{transformer_prefix}final_layer.adaLN_modulation.1.bias"
    
    if adaLN_weight_key in state_dict and adaLN_bias_key in state_dict:
        # Extract the tensors
        adaLN_weight = state_dict[adaLN_weight_key]
        adaLN_bias = state_dict[adaLN_bias_key]
        
        # Swap scale and shift
        half_size = adaLN_weight.shape[0] // 2
        shift_weight, scale_weight = torch.split(adaLN_weight, [half_size, half_size], dim=0)
        shift_bias, scale_bias = torch.split(adaLN_bias, [half_size, half_size], dim=0)
        
        # Combine in the reverse order
        transformer_state_dict["norm_out.linear.weight"] = torch.cat([scale_weight, shift_weight], dim=0)
        transformer_state_dict["norm_out.linear.bias"] = torch.cat([scale_bias, shift_bias], dim=0)
        
        if args.verbose:
            print(f"Processed adaLN modulation with swapped scale and shift")
    
    # Process double_blocks
    for i in range(args.num_layers):
        # Basic tensor mappings
        double_blocks_mapping = {
            f"{transformer_prefix}double_blocks.{i}.img_mod.lin.weight": f"transformer_blocks.{i}.norm1.linear.weight",
            f"{transformer_prefix}double_blocks.{i}.img_mod.lin.bias": f"transformer_blocks.{i}.norm1.linear.bias",
            f"{transformer_prefix}double_blocks.{i}.txt_mod.lin.weight": f"transformer_blocks.{i}.norm1_context.linear.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_mod.lin.bias": f"transformer_blocks.{i}.norm1_context.linear.bias",
            f"{transformer_prefix}double_blocks.{i}.img_attn.norm.query_norm.scale": f"transformer_blocks.{i}.attn.norm_q.weight",
            f"{transformer_prefix}double_blocks.{i}.img_attn.norm.key_norm.scale": f"transformer_blocks.{i}.attn.norm_k.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_attn.norm.query_norm.scale": f"transformer_blocks.{i}.attn.norm_added_q.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_attn.norm.key_norm.scale": f"transformer_blocks.{i}.attn.norm_added_k.weight",
            f"{transformer_prefix}double_blocks.{i}.img_mlp.0.weight": f"transformer_blocks.{i}.ff.net.0.proj.weight",
            f"{transformer_prefix}double_blocks.{i}.img_mlp.0.bias": f"transformer_blocks.{i}.ff.net.0.proj.bias",
            f"{transformer_prefix}double_blocks.{i}.img_mlp.2.weight": f"transformer_blocks.{i}.ff.net.2.weight",
            f"{transformer_prefix}double_blocks.{i}.img_mlp.2.bias": f"transformer_blocks.{i}.ff.net.2.bias",
            f"{transformer_prefix}double_blocks.{i}.txt_mlp.0.weight": f"transformer_blocks.{i}.ff_context.net.0.proj.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_mlp.0.bias": f"transformer_blocks.{i}.ff_context.net.0.proj.bias",
            f"{transformer_prefix}double_blocks.{i}.txt_mlp.2.weight": f"transformer_blocks.{i}.ff_context.net.2.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_mlp.2.bias": f"transformer_blocks.{i}.ff_context.net.2.bias",
            f"{transformer_prefix}double_blocks.{i}.img_attn.proj.weight": f"transformer_blocks.{i}.attn.to_out.0.weight",
            f"{transformer_prefix}double_blocks.{i}.img_attn.proj.bias": f"transformer_blocks.{i}.attn.to_out.0.bias",
            f"{transformer_prefix}double_blocks.{i}.txt_attn.proj.weight": f"transformer_blocks.{i}.attn.to_add_out.weight",
            f"{transformer_prefix}double_blocks.{i}.txt_attn.proj.bias": f"transformer_blocks.{i}.attn.to_add_out.bias",
        }
        
        for all_in_one_key, diffusers_key in double_blocks_mapping.items():
            if all_in_one_key in state_dict:
                transformer_state_dict[diffusers_key] = state_dict[all_in_one_key]
                if args.verbose:
                    print(f"Extracted: {all_in_one_key} -> {diffusers_key}")
        
        # Process img_attn.qkv - split into q, k, v
        img_qkv_key = f"{transformer_prefix}double_blocks.{i}.img_attn.qkv.weight"
        img_qkv_bias_key = f"{transformer_prefix}double_blocks.{i}.img_attn.qkv.bias"
        
        if img_qkv_key in state_dict and img_qkv_bias_key in state_dict:
            # Split into q, k, v - equal three-way split
            q_weight, k_weight, v_weight = torch.chunk(state_dict[img_qkv_key], 3, dim=0)
            q_bias, k_bias, v_bias = torch.chunk(state_dict[img_qkv_bias_key], 3, dim=0)
            
            # Save to transformer state dict
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = q_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = k_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = v_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_q.bias"] = q_bias
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_k.bias"] = k_bias
            transformer_state_dict[f"transformer_blocks.{i}.attn.to_v.bias"] = v_bias
            
            if args.verbose:
                print(f"Split: {img_qkv_key} -> q, k, v with shapes: {q_weight.shape}, {k_weight.shape}, {v_weight.shape}")
        
        # Process txt_attn.qkv - split into q, k, v
        txt_qkv_key = f"{transformer_prefix}double_blocks.{i}.txt_attn.qkv.weight"
        txt_qkv_bias_key = f"{transformer_prefix}double_blocks.{i}.txt_attn.qkv.bias"
        
        if txt_qkv_key in state_dict and txt_qkv_bias_key in state_dict:
            # Split into q, k, v - equal three-way split
            q_weight, k_weight, v_weight = torch.chunk(state_dict[txt_qkv_key], 3, dim=0)
            q_bias, k_bias, v_bias = torch.chunk(state_dict[txt_qkv_bias_key], 3, dim=0)
            
            # Save to transformer state dict
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = q_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = k_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = v_weight
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.bias"] = q_bias
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.bias"] = k_bias
            transformer_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.bias"] = v_bias
            
            if args.verbose:
                print(f"Split: {txt_qkv_key} -> q, k, v with shapes: {q_weight.shape}, {k_weight.shape}, {v_weight.shape}")
    
    # Process single_blocks
    # Calculate the inner_dim from attention parameters
    inner_dim = args.num_attention_heads * args.attention_head_dim
    mlp_ratio = 4.0  # Standard value for Flux models
    mlp_hidden_dim = int(inner_dim * mlp_ratio)
    
    if args.verbose:
        print(f"Using inner_dim: {inner_dim}, mlp_hidden_dim: {mlp_hidden_dim} for splitting single blocks")
    
    for i in range(args.num_single_layers):
        # Basic tensor mappings
        single_blocks_mapping = {
            f"{transformer_prefix}single_blocks.{i}.modulation.lin.weight": f"single_transformer_blocks.{i}.norm.linear.weight",
            f"{transformer_prefix}single_blocks.{i}.modulation.lin.bias": f"single_transformer_blocks.{i}.norm.linear.bias",
            f"{transformer_prefix}single_blocks.{i}.norm.query_norm.scale": f"single_transformer_blocks.{i}.attn.norm_q.weight",
            f"{transformer_prefix}single_blocks.{i}.norm.key_norm.scale": f"single_transformer_blocks.{i}.attn.norm_k.weight",
            f"{transformer_prefix}single_blocks.{i}.linear2.weight": f"single_transformer_blocks.{i}.proj_out.weight",
            f"{transformer_prefix}single_blocks.{i}.linear2.bias": f"single_transformer_blocks.{i}.proj_out.bias",
        }
        
        for all_in_one_key, diffusers_key in single_blocks_mapping.items():
            if all_in_one_key in state_dict:
                transformer_state_dict[diffusers_key] = state_dict[all_in_one_key]
                if args.verbose:
                    print(f"Extracted: {all_in_one_key} -> {diffusers_key}")
        
        # Process linear1 - split into q, k, v, mlp
        linear1_weight_key = f"{transformer_prefix}single_blocks.{i}.linear1.weight"
        linear1_bias_key = f"{transformer_prefix}single_blocks.{i}.linear1.bias"
        
        if linear1_weight_key in state_dict and linear1_bias_key in state_dict:
            # Get the tensors
            linear1_weight = state_dict[linear1_weight_key]
            linear1_bias = state_dict[linear1_bias_key]
            
            # Define the expected split sizes using inner_dim and mlp_ratio
            expected_split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
            expected_total_size = sum(expected_split_size)
            
            # Check the actual tensor size
            actual_size = linear1_weight.shape[0]
            
            if args.verbose:
                print(f"Block {i} linear1:")
                print(f"  Expected split sizes: {expected_split_size}, total: {expected_total_size}")
                print(f"  Actual size: {actual_size}")
            
            # Determine how to split based on the actual size
            if actual_size == expected_total_size:
                # If sizes match, use the standard split
                q_weight, k_weight, v_weight, mlp_weight = torch.split(linear1_weight, expected_split_size, dim=0)
                q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, expected_split_size, dim=0)
                
                if args.verbose:
                    print(f"  Using standard split sizes: {expected_split_size}")
            else:
                # If sizes don't match, check if it's an equal-split model
                if actual_size % 4 == 0:
                    # This seems to be an equal-split model
                    equal_split_size = actual_size // 4
                    split_size = (equal_split_size, equal_split_size, equal_split_size, equal_split_size)
                    
                    q_weight, k_weight, v_weight, mlp_weight = torch.split(linear1_weight, split_size, dim=0)
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
                    
                    if args.verbose:
                        print(f"  Using equal split sizes: {split_size}")
                else:
                    # Not divisible by 4, use proportional splits
                    ratio = actual_size / expected_total_size
                    adjusted_q = int(inner_dim * ratio)
                    adjusted_k = int(inner_dim * ratio)
                    adjusted_v = int(inner_dim * ratio)
                    adjusted_mlp = actual_size - (adjusted_q + adjusted_k + adjusted_v)
                    split_size = (adjusted_q, adjusted_k, adjusted_v, adjusted_mlp)
                    
                    q_weight, k_weight, v_weight, mlp_weight = torch.split(linear1_weight, split_size, dim=0)
                    q_bias, k_bias, v_bias, mlp_bias = torch.split(linear1_bias, split_size, dim=0)
                    
                    if args.verbose:
                        print(f"  Using proportional split sizes: {split_size}")
            
            # Save to transformer state dict
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_q.weight"] = q_weight
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_k.weight"] = k_weight
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_v.weight"] = v_weight
            transformer_state_dict[f"single_transformer_blocks.{i}.proj_mlp.weight"] = mlp_weight
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_q.bias"] = q_bias
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_k.bias"] = k_bias
            transformer_state_dict[f"single_transformer_blocks.{i}.attn.to_v.bias"] = v_bias
            transformer_state_dict[f"single_transformer_blocks.{i}.proj_mlp.bias"] = mlp_bias
            
            if args.verbose:
                print(f"  Final shapes: q: {q_weight.shape}, k: {k_weight.shape}, v: {v_weight.shape}, mlp: {mlp_weight.shape}")
    
    # Get file structure from reference model if provided
    weight_map = {}
    if args.reference_model_path:
        reference_index_path = os.path.join(args.reference_model_path, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(reference_index_path):
            print(f"Using file structure from {reference_index_path}")
            with open(reference_index_path, 'r') as f:
                reference_index = json.load(f)
                weight_map = reference_index.get("weight_map", {})
    
    # If no reference model or invalid reference, create a simple file structure
    if not weight_map:
        print("No valid reference model found, creating a default file structure")
        # Split the weights across two files
        keys = list(transformer_state_dict.keys())
        half_point = len(keys) // 2
        
        # Create a simple weight map
        weight_map = {}
        for i, key in enumerate(keys):
            if i < half_point:
                weight_map[key] = "diffusion_pytorch_model-00001-of-00002.safetensors"
            else:
                weight_map[key] = "diffusion_pytorch_model-00002-of-00002.safetensors"
    
    # Distribute tensors to files according to the weight map
    file_tensors = {}
    for key, file_name in weight_map.items():
        if key in transformer_state_dict:
            if file_name not in file_tensors:
                file_tensors[file_name] = {}
            file_tensors[file_name][key] = transformer_state_dict[key]
    
    # Save files
    print(f"Saving model files to {args.output_path}")
    for file_name, tensors in file_tensors.items():
        if tensors:
            output_file = os.path.join(args.output_path, file_name)
            print(f"Saving {len(tensors)} tensors to {output_file}")
            safetensors.torch.save_file(tensors, output_file)
    
    # Create and save index.json
    # Filter weight_map to only include keys we actually have
    filtered_weight_map = {k: v for k, v in weight_map.items() if k in transformer_state_dict}
    
    index_data = {
        "metadata": {
            "total_size": sum(tensor.nelement() * tensor.element_size() for tensor in transformer_state_dict.values())
        },
        "weight_map": filtered_weight_map
    }
    
    index_file = os.path.join(args.output_path, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"Saved index file to {index_file}")

    # Get the actual joint_attention_dim from the context_embedder.weight tensor shape
    joint_attention_dim = args.num_attention_heads * args.attention_head_dim  # Default calculation
    
    # Override with actual dimension if available
    if "context_embedder.weight" in transformer_state_dict:
        actual_dim = transformer_state_dict["context_embedder.weight"].shape[1]
        joint_attention_dim = actual_dim  # Use the actual dimension from the tensor
        if args.verbose:
            print(f"Using actual joint_attention_dim: {joint_attention_dim} from context_embedder.weight tensor")
    
    # Create and save config.json
    config = {
        "_class_name": "FluxTransformer2DModel",
        "_diffusers_version": "0.32.0.dev0",
        "_name_or_path": "ostris/Flex.1-alpha/transformer",
        "attention_head_dim": args.attention_head_dim,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": True,
        "in_channels": 64,
        "joint_attention_dim": joint_attention_dim,  # Use the detected dimension
        "num_attention_heads": args.num_attention_heads,
        "num_layers": args.num_layers,
        "num_single_layers": args.num_single_layers,
        "patch_size": 1,
        "pooled_projection_dim": 768,
    }
    
    config_file = os.path.join(args.output_path, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_file}")
    
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()