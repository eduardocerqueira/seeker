#date: 2024-11-08T17:04:08Z
#url: https://api.github.com/gists/746b92bfc9c0be38d3f7de919f8ad25e
#owner: https://api.github.com/users/akkypat6234

import torch
import torch.nn as nn
from typing import List, Tensor, Optional

class StackedCrossNet(nn.Module):
    """
    Stacked version of Cross Network (DCN-V2)
    
    Args:
        input_dim: Input feature dimension
        num_layers: Number of cross layers
        layer_width: Width of each cross layer
    """
    def __init__(self, input_dim: int, num_layers: int = 3, layer_width: int = 128):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, input_dim)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        for i in range(self.num_layers):
            xw = self.layers[i](x)
            x = x0 * xw + x
        return x

class GatedCrossNet(nn.Module):
    """
    Gated version of Cross Network (G-DCN)
    
    Args:
        input_dim: Input feature dimension
        num_layers: Number of cross layers
        layer_width: Width of each cross layer
    """
    def __init__(self, input_dim: int, num_layers: int = 3, layer_width: int = 128):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, input_dim)
            ) for _ in range(num_layers)
        ])
        
        # Gating networks
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, layer_width),
                nn.ReLU(),
                nn.Linear(layer_width, input_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        for i in range(self.num_layers):
            # Calculate cross transform
            xw = self.layers[i](x)
            cross_term = x0 * xw
            
            # Calculate gate values
            gate = self.gate_networks[i](x)
            
            # Apply gating mechanism
            x = gate * cross_term + (1 - gate) * x
        return x

class DCNv2ContextHead(nn.Module):
    """
    Stacked DCN-V2 style context head
    
    Args:
        deep_dims: Dimensions of categorical features
        deep_embed_dims: Embedding dimensions
        num_wide: Number of numerical features
        wad_embed_dim: Output embedding dimension
        num_cross_layers: Number of cross network layers
        layer_width: Width of cross layers
    """
    def __init__(self, deep_dims, item_embedding=None, deep_embed_dims=100, 
                 num_wide=0, wad_embed_dim=64, num_shared=1,
                 shared_embeddings_weight=None, num_cross_layers=3, 
                 layer_width=128):
        super().__init__()
        self.num_wide = num_wide
        self.num_deep = len(deep_dims)
        
        # Embedding handling
        if isinstance(deep_embed_dims, int):
            if shared_embeddings_weight is None:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(dim, deep_embed_dims) for dim in deep_dims
                ])
            else:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(dim, deep_embed_dims) 
                    for dim in deep_dims[:-len(shared_embeddings_weight)]
                ])
                from_pretrained = nn.ModuleList([
                    nn.Embedding.from_pretrained(weight, freeze=True)
                    for weight in shared_embeddings_weight
                ])
                self.deep_embedding.extend(from_pretrained)
        
        # Shared embeddings
        if item_embedding and num_shared:
            self.shared_embed = nn.ModuleList([item_embedding] * num_shared)
        else:
            self.shared_embed = None
        
        # Numerical features processing
        if num_wide > 0:
            self.num_batch_norm = nn.BatchNorm1d(num_wide)
            self.wide_projection = nn.Linear(num_wide, deep_embed_dims)
        
        total_input_dim = deep_embed_dims * (self.num_deep + num_shared)
        
        # Stacked Cross Network
        self.cross_net = StackedCrossNet(
            total_input_dim, num_cross_layers, layer_width
        )
        
        # Deep Network
        self.deep_net = nn.Sequential(
            nn.Linear(total_input_dim, wad_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(wad_embed_dim * 2, wad_embed_dim)
        )
        
        # Final projection
        self.final_projection = nn.Linear(total_input_dim + wad_embed_dim, wad_embed_dim)
        self.layer_norm = nn.LayerNorm(wad_embed_dim)
        
    def forward(self, deep_in: List[Tensor], wide_in: Optional[List[Tensor]] = None, 
                shared_in: Optional[List[Tensor]] = None) -> Tensor:
        # Process categorical features
        deep_embeddings = [
            self.deep_embedding[i](input_deep) 
            for i, input_deep in enumerate(deep_in)
        ]
        
        # Add shared embeddings if present
        if shared_in is not None and self.shared_embed is not None:
            shared_embeds = [
                self.shared_embed[i](input_shared)
                for i, input_shared in enumerate(shared_in)
            ]
            deep_embeddings.extend(shared_embeds)
        
        # Process numerical features
        if self.num_wide and wide_in is not None:
            wide_tensor = torch.stack([w.float() for w in wide_in], dim=1)
            wide_normalized = self.num_batch_norm(wide_tensor)
            wide_embedded = self.wide_projection(wide_normalized)
            deep_embeddings.append(wide_embedded)
        
        # Combine embeddings
        combined = torch.cat(deep_embeddings, dim=1)
        
        # Cross Network path
        cross_output = self.cross_net(combined)
        
        # Deep Network path
        deep_output = self.deep_net(combined)
        
        # Combine both paths
        final_input = torch.cat([cross_output, deep_output], dim=1)
        output = self.final_projection(final_input)
        output = self.layer_norm(output)
        
        return output

class GDCNContextHead(nn.Module):
    """
    Gated DCN (G-DCN) style context head
    
    Args:
        Similar to DCNv2ContextHead but uses gated cross layers
    """
    def __init__(self, deep_dims, item_embedding=None, deep_embed_dims=100, 
                 num_wide=0, wad_embed_dim=64, num_shared=1,
                 shared_embeddings_weight=None, num_cross_layers=3, 
                 layer_width=128):
        super().__init__()
        # Similar initialization as DCNv2ContextHead
        self.num_wide = num_wide
        self.num_deep = len(deep_dims)
        
        # Initialize embeddings (same as DCNv2)
        if isinstance(deep_embed_dims, int):
            if shared_embeddings_weight is None:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(dim, deep_embed_dims) for dim in deep_dims
                ])
            else:
                self.deep_embedding = nn.ModuleList([
                    nn.Embedding(dim, deep_embed_dims) 
                    for dim in deep_dims[:-len(shared_embeddings_weight)]
                ])
                from_pretrained = nn.ModuleList([
                    nn.Embedding.from_pretrained(weight, freeze=True)
                    for weight in shared_embeddings_weight
                ])
                self.deep_embedding.extend(from_pretrained)
        
        # Shared embeddings
        if item_embedding and num_shared:
            self.shared_embed = nn.ModuleList([item_embedding] * num_shared)
        else:
            self.shared_embed = None
        
        # Numerical features processing
        if num_wide > 0:
            self.num_batch_norm = nn.BatchNorm1d(num_wide)
            self.wide_projection = nn.Linear(num_wide, deep_embed_dims)
        
        total_input_dim = deep_embed_dims * (self.num_deep + num_shared)
        
        # Replace CrossNet with GatedCrossNet
        self.cross_net = GatedCrossNet(
            total_input_dim, num_cross_layers, layer_width
        )
        
        # Deep Network remains the same
        self.deep_net = nn.Sequential(
            nn.Linear(total_input_dim, wad_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(wad_embed_dim * 2, wad_embed_dim)
        )
        
        # Output processing
        self.final_projection = nn.Linear(total_input_dim + wad_embed_dim, wad_embed_dim)
        self.layer_norm = nn.LayerNorm(wad_embed_dim)
        
    def forward(self, deep_in: List[Tensor], wide_in: Optional[List[Tensor]] = None, 
                shared_in: Optional[List[Tensor]] = None) -> Tensor:
        # Forward pass implementation same as DCNv2ContextHead
        # Process categorical features
        deep_embeddings = [
            self.deep_embedding[i](input_deep) 
            for i, input_deep in enumerate(deep_in)
        ]
        
        if shared_in is not None and self.shared_embed is not None:
            shared_embeds = [
                self.shared_embed[i](input_shared)
                for i, input_shared in enumerate(shared_in)
            ]
            deep_embeddings.extend(shared_embeds)
        
        if self.num_wide and wide_in is not None:
            wide_tensor = torch.stack([w.float() for w in wide_in], dim=1)
            wide_normalized = self.num_batch_norm(wide_tensor)
            wide_embedded = self.wide_projection(wide_normalized)
            deep_embeddings.append(wide_embedded)
        
        combined = torch.cat(deep_embeddings, dim=1)
        
        # Use gated cross network
        cross_output = self.cross_net(combined)
        deep_output = self.deep_net(combined)
        
        final_input = torch.cat([cross_output, deep_output], dim=1)
        output = self.final_projection(final_input)
        output = self.layer_norm(output)
        
        return output