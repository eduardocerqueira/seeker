#date: 2025-07-21T16:42:26Z
#url: https://api.github.com/gists/ecbad3ea1cefcea0afd6b9fc6d7675ca
#owner: https://api.github.com/users/DSamuelHodge

# === Imports ===
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import gudhi as gd
from sklearn.manifold import MDS
import warnings

warnings.filterwarnings("ignore")

# === Constants and Configurations ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT_TEXT = "Topology reveals the hidden structure of attention patterns."
MAX_TOKENS = "**********"
TARGET_LAYERS = [2, 8, 16, 21, 23]
SELECTED_HEAD_POLICY = "middle"  # Can extend to "first", "last", or custom
EPSILONS = [0.3, 0.5, 0.8]
PERSISTENCE_MAX_EDGE = 1.0
RANDOM_SEED = 42


# === Helper Functions ===

 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"a "**********"d "**********"_ "**********"q "**********"w "**********"e "**********"n "**********"_ "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"a "**********"n "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********"_ "**********"n "**********"a "**********"m "**********"e "**********") "**********": "**********"
    """Load Qwen model and tokenizer with correct trust and fallback token setup."""
    tokenizer = "**********"=True)
    model = AutoModel.from_pretrained(model_name, output_attentions=True, trust_remote_code=True, attn_implementation="eager")

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"p "**********"a "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        tokenizer.pad_token = "**********"
    return tokenizer, model


 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"x "**********"t "**********"r "**********"a "**********"c "**********"t "**********"_ "**********"a "**********"t "**********"t "**********"e "**********"n "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"t "**********"e "**********"x "**********"t "**********"( "**********"m "**********"o "**********"d "**********"e "**********"l "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********", "**********"  "**********"t "**********"e "**********"x "**********"t "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********") "**********": "**********"
    """Tokenize input and extract attention tensors from Qwen2.5-0.5B."""
    inputs = "**********"="pt", max_length=max_tokens, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    if outputs.attentions is None:
        raise ValueError("No attention returned by model.")

    return torch.stack([a for a in outputs.attentions if a is not None]).squeeze(1).detach().cpu()


def compute_embedding_from_attention(attn_matrix):
    """Symmetrize attention into a distance matrix and embed with MDS."""
    attn = attn_matrix.numpy()
    dist = 1.0 - (attn + attn.T) / 2.0
    return MDS(n_components=2, dissimilarity='precomputed', random_state=RANDOM_SEED).fit_transform(dist)


def create_rips_complex(points, epsilon, max_dim=2):
    """Construct Rips complex and return its simplex tree."""
    rips = gd.RipsComplex(points=points, max_edge_length=epsilon)
    return rips.create_simplex_tree(max_dimension=max_dim)


def draw_simplicial_complex(ax, points, simplex_tree, epsilon):
    """Render 2D simplicial complex given simplex tree and embedding."""
    vertices, edges, triangles = [], [], []
    for simplex, _ in simplex_tree.get_simplices():
        if len(simplex) == 1:
            vertices.append(simplex[0])
        elif len(simplex) == 2:
            edges.append(simplex)
        elif len(simplex) == 3:
            triangles.append(simplex)

    for triangle in triangles:
        if all(v < len(points) for v in triangle):
            poly = plt.Polygon(points[triangle], alpha=0.3, facecolor='lightblue', edgecolor='blue', linewidth=1)
            ax.add_patch(poly)

    for edge in edges:
        if all(v < len(points) for v in edge):
            ax.plot(*points[edge].T, 'b-', linewidth=2, alpha=0.8)

    ax.scatter(points[:, 0], points[:, 1], c='darkblue', s=100, zorder=5)
    ax.set_title(f"Simplicial Complex (ε = {epsilon:.1f})", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(True, alpha=0.3)


def create_topological_visualization(attn_matrix, layer_idx):
    """Create professional visualization of attention topology and persistent homology."""
    points = compute_embedding_from_attention(attn_matrix)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Qwen2.5-0.5B Layer {layer_idx} - Topological Analysis", fontsize=16, weight='bold')

    for i, eps in enumerate(EPSILONS):
        ax = fig.add_subplot(2, 3, i + 1)
        st = create_rips_complex(points, eps)
        draw_simplicial_complex(ax, points, st, eps)

    # Persistence diagram (bottom row)
    ax_persist = fig.add_subplot(2, 3, (4, 6))
    st_full = create_rips_complex(points, PERSISTENCE_MAX_EDGE)
    persistence = st_full.persistence()
    gd.plot_persistence_diagram(persistence, axes=ax_persist)
    ax_persist.set_title("Persistence Diagram", fontsize=14)

    plt.tight_layout()
    return fig


# === Main Execution ===

def main():
    tokenizer, model = "**********"
    attention = "**********"

    for layer in TARGET_LAYERS:
        if layer >= attention.shape[0]:
            continue
        head_idx = attention.shape[1] // 2  # Default: middle head
        attn_matrix = attention[layer, head_idx]
        fig = create_topological_visualization(attn_matrix, layer)
        fig.savefig(f'qwen_layer_{layer}_topology.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: Layer {layer} topology visual")

    print("All visualizations complete.")

if __name__ == "__main__":
    main()
