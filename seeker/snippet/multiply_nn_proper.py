#date: 2026-02-24T17:41:07Z
#url: https://api.github.com/gists/24fd745905c9c3c2f17deab6910156d6
#owner: https://api.github.com/users/secemp9

#!/usr/bin/env python3
"""
43-Parameter MLP for Multiplication — PyTorch Port

Original solution by Stefan Mesken (Code Golf, 2019)
Ported from TensorFlow to PyTorch with identical weights.

A standard neural network architecture with weights that would emerge
from actual training — not hand-crafted mathematical tricks.

Architecture: Input(2) → Dense(6, tanh) → Dense(3, tanh) → Dense(1, linear)
Parameters:  43 (18 + 21 + 4)
Max error:   ~0.44 (within the <0.5 requirement)

Original: https://codegolf.stackexchange.com/a/187605
"""

import torch
import torch.nn as nn


class MultiplicationMLP(nn.Module):
    """
    A standard 3-layer MLP trained to approximate multiplication.

    This is a completely normal neural network architecture - the kind you'd
    use for any regression task. The only special thing is that it was trained
    specifically to output x1 * x2 given inputs [x1, x2].

    The network learned to approximate multiplication through standard
    backpropagation and gradient descent, NOT through any mathematical tricks.
    """

    def __init__(self):
        super().__init__()

        # Standard MLP layers - nothing special about the architecture
        self.layer1 = nn.Linear(in_features=2, out_features=6)   # 2*6 + 6 = 18 params
        self.layer2 = nn.Linear(in_features=6, out_features=3)   # 6*3 + 3 = 21 params
        self.layer3 = nn.Linear(in_features=3, out_features=1)   # 3*1 + 1 = 4 params
        # Total: 43 parameters

        # Load the TRAINED weights from Stefan Mesken's solution
        self._load_trained_weights()

    def _load_trained_weights(self):
        """
        Load weights that were obtained through ACTUAL TRAINING.

        These are not hand-crafted or mathematically derived - they are the
        result of gradient descent optimization on a multiplication dataset.

        Note: PyTorch stores Linear weights as (out_features, in_features),
        so we transpose from the mathematical notation (in_features, out_features).
        """
        with torch.no_grad():
            # Layer 1: Input (2) -> Hidden1 (6)
            # Original weights given as (2x6), transposed for PyTorch (6x2)
            layer1_weights = torch.tensor([
                [ 0.10697944,  0.05394982,  0.05479664, -0.04538541,  0.05369904, -0.0728976 ],
                [ 0.10571832,  0.05576797, -0.04670485, -0.04466859, -0.05855528, -0.07390639]
            ], dtype=torch.float32).T  # Transpose to (6, 2)

            self.layer1.weight.copy_(layer1_weights)

            self.layer1.bias.copy_(torch.tensor([
                -3.4242163, -0.8875816, -1.7694025, -1.9409281, 1.7825342, 1.1364107
            ], dtype=torch.float32))

            # Layer 2: Hidden1 (6) -> Hidden2 (3)
            # Original weights given as (6x3), transposed for PyTorch (3x6)
            layer2_weights = torch.tensor([
                [-3.0665843 ,  0.64912266,  3.7107112 ],
                [ 0.4914808 ,  2.1569328 ,  0.65417236],
                [ 3.461693  ,  1.2072319 , -4.181983  ],
                [-2.8746269 , -4.9959164 ,  4.505049  ],
                [-2.920127  , -0.0665407 ,  4.1409926 ],
                [ 1.3777553 , -3.3750365 , -0.10507642]
            ], dtype=torch.float32).T  # Transpose to (3, 6)

            self.layer2.weight.copy_(layer2_weights)

            self.layer2.bias.copy_(torch.tensor([
                -1.376577, 2.8885336, 0.19852689
            ], dtype=torch.float32))

            # Layer 3: Hidden2 (3) -> Output (1)
            # Original weights given as (3x1), transposed for PyTorch (1x3)
            layer3_weights = torch.tensor([
                [-78.7569],
                [-23.602606],
                [84.29587]
            ], dtype=torch.float32).T  # Transpose to (1, 3)

            self.layer3.weight.copy_(layer3_weights)

            self.layer3.bias.copy_(torch.tensor([
                8.521169
            ], dtype=torch.float32))

    def forward(self, x):
        """
        Standard MLP forward pass with tanh activations.

        Args:
            x: Tensor of shape (batch, 2) containing [x1, x2] pairs

        Returns:
            Tensor of shape (batch, 1) approximating x1 * x2
        """
        # Hidden layer 1 with tanh activation
        x = torch.tanh(self.layer1(x))

        # Hidden layer 2 with tanh activation
        x = torch.tanh(self.layer2(x))

        # Output layer (linear - no activation)
        x = self.layer3(x)

        return x


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("MULTIPLICATION NEURAL NETWORK")
    print("Stefan Mesken's 43-Weight Trained Solution")
    print("=" * 70)
    print()

    print("ARCHITECTURE:")
    print("  This is a STANDARD MLP that was TRAINED to approximate multiplication.")
    print("  It is NOT a mathematical trick - these weights emerged from training.")
    print()
    print("  Layer 1: Linear(2 -> 6) + tanh    [18 parameters]")
    print("  Layer 2: Linear(6 -> 3) + tanh    [21 parameters]")
    print("  Layer 3: Linear(3 -> 1) + linear  [4 parameters]")
    print("-" * 70)

    # Create the model
    model = MultiplicationMLP()
    model.eval()

    # Count parameters
    param_count = count_parameters(model)
    print(f"\nTotal Parameters: {param_count}")
    assert param_count == 43, f"Expected 43 parameters, got {param_count}"
    print("✓ Parameter count verified: 43")

    # Test on ALL integer pairs from -10 to 10
    print("\n" + "-" * 70)
    print("TESTING: All integer pairs from -10 to 10")
    print("-" * 70)

    test_range = range(-10, 11)  # -10 to 10 inclusive
    test_cases = []

    for x1 in test_range:
        for x2 in test_range:
            test_cases.append((x1, x2))

    print(f"Number of test cases: {len(test_cases)}")

    # Create batch input
    inputs = torch.tensor(test_cases, dtype=torch.float32)

    # Get predictions
    with torch.no_grad():
        predictions = model(inputs).squeeze()

    # Calculate expected values
    expected = torch.tensor([x1 * x2 for x1, x2 in test_cases], dtype=torch.float32)

    # Calculate deviations
    deviations = torch.abs(predictions - expected)
    max_deviation = deviations.max().item()
    mean_deviation = deviations.mean().item()

    # Find worst case
    worst_idx = deviations.argmax().item()
    worst_x1, worst_x2 = test_cases[worst_idx]
    worst_pred = predictions[worst_idx].item()
    worst_expected = expected[worst_idx].item()

    print(f"\nResults:")
    print(f"  Max deviation:  {max_deviation:.6f}")
    print(f"  Mean deviation: {mean_deviation:.6f}")
    print(f"\nWorst case:")
    print(f"  Input: ({worst_x1}, {worst_x2})")
    print(f"  Expected: {worst_expected:.0f}")
    print(f"  Predicted: {worst_pred:.6f}")
    print(f"  Error: {abs(worst_pred - worst_expected):.6f}")

    # Verify max deviation is acceptable
    print("\n" + "-" * 70)
    if max_deviation < 0.5:
        print(f"✓ SUCCESS: Max deviation ({max_deviation:.6f}) < 0.5")
    else:
        print(f"✗ FAILURE: Max deviation ({max_deviation:.6f}) >= 0.5")

    # Show some example predictions
    print("\n" + "-" * 70)
    print("SAMPLE PREDICTIONS:")
    print("-" * 70)
    print(f"{'Input':^15} {'Expected':^12} {'Predicted':^12} {'Error':^10}")
    print("-" * 70)

    sample_pairs = [
        (0, 0), (1, 1), (2, 3), (5, 5), (-3, 4),
        (7, -8), (-9, -9), (10, 10), (-10, 10), (6, 7)
    ]

    for x1, x2 in sample_pairs:
        inp = torch.tensor([[x1, x2]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(inp).item()
        exp = x1 * x2
        err = abs(pred - exp)
        print(f"({x1:3d}, {x2:3d})      {exp:8.0f}     {pred:10.4f}   {err:8.4f}")

    print("\n" + "=" * 70)
    print("This demonstrates that a standard neural network CAN learn to multiply!")
    print("The weights above are the result of actual training, not manual design.")
    print("=" * 70)


if __name__ == "__main__":
    main()
