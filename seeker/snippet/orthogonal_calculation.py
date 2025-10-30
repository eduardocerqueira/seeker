#date: 2025-10-30T17:12:16Z
#url: https://api.github.com/gists/add58b3cd4907c0dccb752a4ca18efcd
#owner: https://api.github.com/users/adi-kmt

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile
from huggingface_hub import hf_hub_download

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.json"):
    """Load model by downloading only the necessary files"""
    print(f"Downloading model files from {repo_id}...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=tmpdir)
        config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=tmpdir)
        
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        print("Config loaded:", config_dict)
        
        cfg = GPTConfig(
            block_size=config_dict['block_size'],
            vocab_size=config_dict['vocab_size'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embed=config_dict['n_embd'],
            dropout=config_dict.get('dropout', 0.0),
        )
        
        model = GPT(cfg)
        state_dict = torch.load(model_path, map_location='cpu')
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        print(f"Loaded model: {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embed} embed dim")
        return model, cfg


# =============================================================================
# METHOD 1: Projection Weight Similarity (Your Original Method)
# =============================================================================
@torch.no_grad()
def compute_weight_similarity(heads, n_head):
    """
    Compute cosine similarity between projection weight matrices.
    This measures if the raw weights are similar.
    heads: [n_head, head_dim, n_embed]
    """
    similarity_matrix = torch.zeros(n_head, n_head)
    
    for i in range(n_head):
        for j in range(n_head):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                vec_i = heads[i].flatten()
                vec_j = heads[j].flatten()
                cosine_sim = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
                similarity_matrix[i, j] = cosine_sim.item()
    
    return similarity_matrix


# =============================================================================
# METHOD 2: Subspace Orthogonality (Better Mathematical Measure)
# =============================================================================
@torch.no_grad()
def compute_subspace_orthogonality(heads, n_head):
    """
    Measure orthogonality of the subspaces each head projects to.
    Uses normalized Frobenius inner product.
    heads: [n_head, head_dim, n_embed]
    
    Returns similarity matrix where:
    - 1.0 means heads project to identical subspaces
    - 0.0 means heads project to orthogonal subspaces
    - -1.0 means heads project to opposite subspaces
    """
    similarity_matrix = torch.zeros(n_head, n_head)
    
    for i in range(n_head):
        for j in range(n_head):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                W_i = heads[i]  # [head_dim, n_embed]
                W_j = heads[j]  # [head_dim, n_embed]
                
                # Frobenius inner product: sum of element-wise products
                inner_prod = torch.sum(W_i * W_j)
                
                # Normalize by Frobenius norms
                norm_i = torch.norm(W_i, p='fro')
                norm_j = torch.norm(W_j, p='fro')
                
                sim = (inner_prod / (norm_i * norm_j + 1e-8)).item()
                similarity_matrix[i, j] = sim
    
    return similarity_matrix


# =============================================================================
# METHOD 3: Functional Orthogonality (Best - Uses Actual Data)
# =============================================================================
@torch.no_grad()
 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"o "**********"m "**********"p "**********"u "**********"t "**********"e "**********"_ "**********"f "**********"u "**********"n "**********"c "**********"t "**********"i "**********"o "**********"n "**********"a "**********"l "**********"_ "**********"s "**********"i "**********"m "**********"i "**********"l "**********"a "**********"r "**********"i "**********"t "**********"y "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"l "**********"a "**********"y "**********"e "**********"r "**********"_ "**********"i "**********"d "**********"x "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"n "**********"u "**********"m "**********"_ "**********"s "**********"a "**********"m "**********"p "**********"l "**********"e "**********"s "**********"= "**********"1 "**********"0 "**********"0 "**********", "**********"  "**********"s "**********"e "**********"q "**********"_ "**********"l "**********"e "**********"n "**********"= "**********"1 "**********"2 "**********"8 "**********") "**********": "**********"
    """
    Measure functional similarity by analyzing attention patterns on actual data.
    This is the most meaningful measure as it shows how heads actually behave.
    
    Returns:
    - attention_pattern_similarity: similarity of attention distributions
    - output_similarity: similarity of head outputs
    """
    device = next(model.parameters()).device
    attn_layer = model.tr.h[layer_idx].attn
    n_head = model.cfg.n_head
    
    # Generate random token sequences
    vocab_size = model.cfg.vocab_size
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    
    # Storage for attention patterns and outputs
    all_attn_patterns = []
    all_head_outputs = []
    
    # Hook to capture attention patterns
    attn_patterns_batch = []
    head_outputs_batch = []
    
    def hook_fn(module, input, output):
        # We need to capture q, k, v and attention weights
        pass
    
    # Run forward pass and manually extract attention for this layer
    with torch.no_grad():
        # Get embeddings up to this layer
        x = model.tr.wte(input_ids)
        x = model.tr.wpe(torch.arange(seq_len, device=device).unsqueeze(0))
        x = model.tr.wte(input_ids) + model.tr.wpe(torch.arange(seq_len, device=device).unsqueeze(0))
        
        # Pass through layers up to target layer
        for i in range(layer_idx):
            x = model.tr.h[i](x)
        
        # Now manually compute attention for the target layer
        B, T, C = x.size()
        
        # Get qkv
        qkv = attn_layer.c_attn(x)
        q, k, v = qkv.split(model.cfg.n_embed, dim=2)
        
        # Reshape to separate heads
        head_dim = C // n_head
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        # Apply head mask if present
        if hasattr(attn_layer, 'head_mask'):
            mask = attn_layer.head_mask.view(1, n_head, 1, 1).to(q.dtype).to(q.device)
            q = q * mask
            k = k * mask
            v = v * mask
        
        # Compute attention weights manually
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_head, T, T]
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute head outputs
        head_outputs = torch.matmul(attn_weights, v)  # [B, n_head, T, head_dim]
        
        # Store patterns (average over batch and sequence for each head)
        # Shape: [n_head, T, T] -> average to [n_head, T*T]
        attn_patterns = attn_weights.mean(dim=0)  # Average over batch: [n_head, T, T]
        attn_patterns_flat = attn_patterns.view(n_head, -1)  # [n_head, T*T]
        
        # Store head outputs (average over batch and sequence)
        # Shape: [B, n_head, T, head_dim] -> [n_head, head_dim]
        head_outputs_avg = head_outputs.mean(dim=(0, 2))  # [n_head, head_dim]
    
    # Compute similarity between attention patterns
    attn_similarity = torch.zeros(n_head, n_head)
    for i in range(n_head):
        for j in range(n_head):
            if i == j:
                attn_similarity[i, j] = 1.0
            else:
                # Cosine similarity of attention distributions
                sim = F.cosine_similarity(
                    attn_patterns_flat[i].unsqueeze(0),
                    attn_patterns_flat[j].unsqueeze(0)
                )
                attn_similarity[i, j] = sim.item()
    
    # Compute similarity between head outputs
    output_similarity = torch.zeros(n_head, n_head)
    for i in range(n_head):
        for j in range(n_head):
            if i == j:
                output_similarity[i, j] = 1.0
            else:
                sim = F.cosine_similarity(
                    head_outputs_avg[i].unsqueeze(0),
                    head_outputs_avg[j].unsqueeze(0)
                )
                output_similarity[i, j] = sim.item()
    
    return attn_similarity, output_similarity


# =============================================================================
# Visualization and Analysis
# =============================================================================
def plot_similarity_heatmap(similarity_matrix, layer_idx, title, save_path=None):
    """Plot heatmap of head similarities"""
    n_head = similarity_matrix.shape[0]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix.cpu().numpy() if torch.is_tensor(similarity_matrix) else similarity_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                xticklabels=range(n_head),
                yticklabels=range(n_head),
                cbar_kws={'label': 'Similarity'})
    
    plt.title(f'Layer {layer_idx} - {title}')
    plt.xlabel('Head Index')
    plt.ylabel('Head Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def compute_statistics(similarity_matrix, n_head):
    """Compute statistics from similarity matrix (excluding diagonal)"""
    mask = ~torch.eye(n_head, dtype=bool)
    off_diagonal = similarity_matrix[mask]
    
    return {
        'avg_similarity': off_diagonal.mean().item(),
        'std_similarity': off_diagonal.std().item(),
        'max_similarity': off_diagonal.max().item(),
        'min_similarity': off_diagonal.min().item(),
        'num_high_similarity': (off_diagonal > 0.8).sum().item(),
        'num_orthogonal': (torch.abs(off_diagonal) < 0.1).sum().item(),
    }


@torch.no_grad()
 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"o "**********"m "**********"p "**********"r "**********"e "**********"h "**********"e "**********"n "**********"s "**********"i "**********"v "**********"e "**********"_ "**********"h "**********"e "**********"a "**********"d "**********"_ "**********"a "**********"n "**********"a "**********"l "**********"y "**********"s "**********"i "**********"s "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"c "**********"f "**********"g "**********", "**********"  "**********"l "**********"a "**********"y "**********"e "**********"r "**********"_ "**********"i "**********"d "**********"x "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
    """
    Run all three types of analysis on a single layer.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS - Layer {layer_idx}")
    print(f"{'='*80}")
    
    attn_layer = model.tr.h[layer_idx].attn
    n_embed = cfg.n_embed
    n_head = cfg.n_head
    head_dim = n_embed // n_head
    
    # Extract projection weights
    qkv_weight = attn_layer.c_attn.weight
    q_proj = qkv_weight[:n_embed, :]
    k_proj = qkv_weight[n_embed:2*n_embed, :]
    v_proj = qkv_weight[2*n_embed:3*n_embed, :]
    
    q_heads = q_proj.view(n_head, head_dim, n_embed)
    k_heads = k_proj.view(n_head, head_dim, n_embed)
    v_heads = v_proj.view(n_head, head_dim, n_embed)
    
    results = {}
    
    # Method 1: Weight Similarity
    print("\n" + "="*80)
    print("METHOD 1: PROJECTION WEIGHT SIMILARITY")
    print("="*80)
    print("Measures: Raw cosine similarity of weight matrices")
    print("Interpretation: How similar are the learned weight parameters?\n")
    
    for proj_name, heads in [('Query', q_heads), ('Key', k_heads), ('Value', v_heads)]:
        print(f"\n--- {proj_name} Weights ---")
        sim_matrix = compute_weight_similarity(heads, n_head)
        stats = compute_statistics(sim_matrix, n_head)
        results[f'{proj_name.lower()}_weight_sim'] = {'matrix': sim_matrix, 'stats': stats}
        
        print(f"Average similarity: {stats['avg_similarity']:.4f} ± {stats['std_similarity']:.4f}")
        print(f"Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
        print(f"High similarity pairs (>0.8): {stats['num_high_similarity']}")
        print(f"Nearly orthogonal pairs (<0.1): {stats['num_orthogonal']}")
        
        plot_similarity_heatmap(
            sim_matrix, layer_idx, 
            f"{proj_name} Weight Similarity (Method 1)",
            f"layer{layer_idx}_{proj_name.lower()}_weight_sim.png"
        )
    
    # Method 2: Subspace Orthogonality
    print("\n" + "="*80)
    print("METHOD 2: SUBSPACE ORTHOGONALITY")
    print("="*80)
    print("Measures: Normalized Frobenius inner product of projection matrices")
    print("Interpretation: How orthogonal are the subspaces heads project to?\n")
    
    for proj_name, heads in [('Query', q_heads), ('Key', k_heads), ('Value', v_heads)]:
        print(f"\n--- {proj_name} Subspace ---")
        sim_matrix = compute_subspace_orthogonality(heads, n_head)
        stats = compute_statistics(sim_matrix, n_head)
        results[f'{proj_name.lower()}_subspace_orth'] = {'matrix': sim_matrix, 'stats': stats}
        
        print(f"Average subspace overlap: {stats['avg_similarity']:.4f} ± {stats['std_similarity']:.4f}")
        print(f"Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
        print(f"High overlap pairs (>0.8): {stats['num_high_similarity']}")
        print(f"Orthogonal subspace pairs (<0.1): {stats['num_orthogonal']}")
        
        plot_similarity_heatmap(
            sim_matrix, layer_idx,
            f"{proj_name} Subspace Orthogonality (Method 2)",
            f"layer{layer_idx}_{proj_name.lower()}_subspace_orth.png"
        )
    
    # Method 3: Functional Similarity
    print("\n" + "="*80)
    print("METHOD 3: FUNCTIONAL SIMILARITY (using actual data)")
    print("="*80)
    print("Measures: Similarity of attention patterns and outputs on real inputs")
    print("Interpretation: How do heads actually behave differently?\n")
    
    attn_sim, output_sim = "**********"
    
    print(f"\n--- Attention Pattern Similarity ---")
    attn_stats = compute_statistics(attn_sim, n_head)
    results['attention_pattern_sim'] = {'matrix': attn_sim, 'stats': attn_stats}
    
    print(f"Average similarity: {attn_stats['avg_similarity']:.4f} ± {attn_stats['std_similarity']:.4f}")
    print(f"Range: [{attn_stats['min_similarity']:.4f}, {attn_stats['max_similarity']:.4f}]")
    print(f"Redundant attention pairs (>0.8): {attn_stats['num_high_similarity']}")
    print(f"Diverse attention pairs (<0.1): {attn_stats['num_orthogonal']}")
    
    plot_similarity_heatmap(
        attn_sim, layer_idx,
        "Attention Pattern Similarity (Method 3)",
        f"layer{layer_idx}_attention_functional.png"
    )
    
    print(f"\n--- Head Output Similarity ---")
    output_stats = compute_statistics(output_sim, n_head)
    results['output_sim'] = {'matrix': output_sim, 'stats': output_stats}
    
    print(f"Average similarity: {output_stats['avg_similarity']:.4f} ± {output_stats['std_similarity']:.4f}")
    print(f"Range: [{output_stats['min_similarity']:.4f}, {output_stats['max_similarity']:.4f}]")
    print(f"Redundant output pairs (>0.8): {output_stats['num_high_similarity']}")
    print(f"Diverse output pairs (<0.1): {output_stats['num_orthogonal']}")
    
    plot_similarity_heatmap(
        output_sim, layer_idx,
        "Head Output Similarity (Method 3)",
        f"layer{layer_idx}_output_functional.png"
    )
    
    return results


def analyze_all_methods(repo_id, layers_to_analyze=None):
    """
    Complete analysis pipeline for all three methods.
    """
    print("="*80)
    print("COMPREHENSIVE ATTENTION HEAD ORTHOGONALITY ANALYSIS")
    print("="*80)
    print("\nThis analysis uses THREE methods to measure head similarity:\n")
    print("1. WEIGHT SIMILARITY: Cosine similarity of raw projection weights")
    print("   - Fast, but least meaningful")
    print("   - Doesn't account for how weights are actually used\n")
    print("2. SUBSPACE ORTHOGONALITY: Normalized Frobenius inner product")
    print("   - Better mathematical measure of projection overlap")
    print("   - Still doesn't require actual data\n")
    print("3. FUNCTIONAL SIMILARITY: Attention patterns on real inputs")
    print("   - Most meaningful - shows actual behavior")
    print("   - Measures both attention distributions and outputs\n")
    print("="*80)
    
    model, cfg = load_custom_model_from_hf(repo_id)
    
    n_layers = cfg.n_layer
    if layers_to_analyze is None:
        layers_to_analyze = [0, n_layers//2, n_layers-1]
    
    print(f"\nAnalyzing layers: {layers_to_analyze}")
    
    all_results = {}
    for layer_idx in layers_to_analyze:
        results = comprehensive_head_analysis(model, cfg, layer_idx)
        all_results[layer_idx] = results
    
    # Print comparison summary
    print("\n" + "="*80)
    print("CROSS-METHOD COMPARISON SUMMARY")
    print("="*80)
    print("\nWhich method shows the most head diversity?\n")
    
    for layer_idx in layers_to_analyze:
        print(f"\nLayer {layer_idx}:")
        print("-" * 60)
        results = all_results[layer_idx]
        
        print(f"Method 1 (Query Weight Sim):      {results['query_weight_sim']['stats']['avg_similarity']:.4f}")
        print(f"Method 2 (Query Subspace Orth):   {results['query_subspace_orth']['stats']['avg_similarity']:.4f}")
        print(f"Method 3 (Attention Pattern):     {results['attention_pattern_sim']['stats']['avg_similarity']:.4f}")
        print(f"Method 3 (Head Output):           {results['output_sim']['stats']['avg_similarity']:.4f}")
        
        print(f"\nOrthogonal pairs (<0.1):")
        print(f"  Method 1: {results['query_weight_sim']['stats']['num_orthogonal']}")
        print(f"  Method 2: {results['query_subspace_orth']['stats']['num_orthogonal']}")
        print(f"  Method 3 (attn): {results['attention_pattern_sim']['stats']['num_orthogonal']}")
        print(f"  Method 3 (output): {results['output_sim']['stats']['num_orthogonal']}")
    
    return all_results


# Usage
if __name__ == "__main__":
    repo_id = "CK0607/fineweb10B-gpt-heads12_L12_E768_max8000_bs128"
    layers = [0, 5, 11]  # First, middle, last
    
    results = analyze_all_methods(repo_id, layers)