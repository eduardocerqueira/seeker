#date: 2025-02-25T17:00:06Z
#url: https://api.github.com/gists/f451fbb75357013dd8d41bd145e38396
#owner: https://api.github.com/users/tam17aki

# -*- coding: utf-8 -*-
"""A demonstration script of Hopfield Network to evaluate dynamics of recall process.

Copyright (C) 2025 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse

import numpy as np
import numpy.typing as npt


def limited_float(x: str) -> float:
    """Convert the input to a float and checks if it's within the range [0.0, 1.0].

    Args:
        x (str): The value to convert and check.

    Returns:
        y (float): The float value if it's within the range.

    Raises:
        argparse.ArgumentTypeError: If the value is not a float or is outside the range.
    """
    try:
        y = float(x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{x} is not a float value") from exc
    if y < 0.0 or y > 1.0:
        raise argparse.ArgumentTypeError(f"{x} is not within the range [0.0, 1.0]")
    return y


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hopfield Network Simulation")
    parser.add_argument(
        "--num_neurons", type=int, default=20, help="Number of neurons (default: 20)"
    )
    parser.add_argument(
        "--num_patterns",
        type=int,
        default=3,
        help="Number of patterns to be learned (default: 3)",
    )
    parser.add_argument(
        "--neuron_threshold",
        type=float,
        default=0.0,
        help="Activation threshold for the neurons (default: 0.0)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="The number of steps for recall process (default: 30)",
    )
    parser.add_argument(
        "--similarity",
        default=0.5,
        type=limited_float,
        help="Similarity of initial pattern (default: 0.5)",
    )
    parser.add_argument(
        "--self_connection",
        type=bool,
        default=False,
        help="Flag to allow self-connection in weights (default: False)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="log.txt",
        help="Filename to save direction cosines (default: log.txt)",
    )
    return parser.parse_args()


def direction_cosine(
    pattern1: npt.NDArray[np.int64], pattern2: npt.NDArray[np.int64]
) -> float:
    """Create a noisy initial state for a given pattern.

    Args:
        pattern1 (npt.NDArray[np.int64]): The original pattern.
        pattern2 (npt.NDArray[np.int64]): The original pattern.

    Returns:
        dircos (float): Direction cosine (similarity).
    """
    num_neurons = len(pattern1)
    distance = np.dot(pattern1, pattern2)
    dircos: float = distance / num_neurons
    return dircos


class HopfieldNetwork:
    """A simple implementation of a Hopfield network using NumPy."""

    def __init__(
        self,
        num_neurons: int,
        neuron_threshold: float = 0.0,
        self_connection: bool = False,
    ):
        """Initialize the Hopfield network.

        Args:
            num_neurons (int): The number of neurons in the network.
            neuron_threshold (float): Activation threshold for the neurons.
                Defaults to 0.0.
            self_conneection (bool): Flag to allow self-connection in weights.
                Defaults to False.
        """
        self.num_neurons: int = num_neurons
        self.weights: npt.NDArray[np.float64] = np.zeros((num_neurons, num_neurons))
        self.neuron_threshold: float = neuron_threshold
        self.self_connection: bool = self_connection

    def learn(self, patterns: npt.NDArray[np.int64]) -> None:
        """Learn the given patterns by Hebbian learning rule.

        Args:
            patterns (npt.NDArray[np.int64]): A 2D NumPy array where each row
                represents a pattern.
        """
        # Adjust the weights by Hebbian learning rule
        num_neurons = patterns.shape[1]
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= num_neurons
        if self.self_connection is False:
            np.fill_diagonal(self.weights, 0.0)

    def recall(
        self,
        initial_state: npt.NDArray[np.int64],
        reference_state: npt.NDArray[np.int64],
        num_steps: int,
    ) -> npt.NDArray[np.float64]:
        """Recall a pattern from the network, starting from the initial state.

        Args:
            initial_state (npt.NDArray[np.int64]): The initial state of the network.
            reference_state (npt.NDArray[np.int64]): The reference state.
            num_steps (int): The number of steps for the recall process.

        Returns:
            dircos_array (npt.NDArray[np.float64]): The array of direction cosines.
        """
        state = initial_state.copy()
        dircos_list = []
        dircos_list.append(direction_cosine(state, reference_state))
        for _ in range(num_steps):
            state = np.sign(self.weights @ state - self.neuron_threshold)
            state[state == 0] = np.random.choice([-1, 1], size=np.sum(state == 0))
            dircos_list.append(direction_cosine(state, reference_state))
        dircos_array = np.array(dircos_list)
        return dircos_array


def define_patterns(num_neurons: int, num_patterns: int) -> npt.NDArray[np.int64]:
    """Define bit patterns to be learned.

    Args:
        num_neurons (int): The number of neurons in the network.
        num_patterns (int): The number of patterns to generate.

    Returns:
        npt.NDArray[np.int64]: A 2D NumPy array where each row represents a pattern.
    """
    patterns: npt.NDArray[np.int64] = (
        np.random.randint(0, 2, (num_patterns, num_neurons), dtype=np.int64) * 2 - 1
    )
    return patterns


def init_state(
    pattern: npt.NDArray[np.int64], num_neurons: int, initial_dircos: float
) -> npt.NDArray[np.int64]:
    """Initialize state vector with direction cosine to original pattern.

    Flips a subset of pattern elements randomly to achieve the desired
    direction cosine. Creates initial states similar to the original
    pattern but with controlled levels of noise.

    Args:
        pattern (npt.NDArray[np.int64]): The original pattern.
        num_neurons (int): The number of neurons in the network.
        initial_dircos (float): Initial direction cosine, determining
            the similarity between the pattern and the initial state.

    Returns:
        initial_state (npt.NDArray[np.int64]): The initial state.
    """
    hamming_dist = (1 - initial_dircos) / 2  # a normalized Hamming distance
    num_to_extract = int(num_neurons * hamming_dist)
    numbers = np.arange(num_neurons)
    np.random.shuffle(numbers)
    initial_state: npt.NDArray[np.int64] = pattern.copy()
    initial_state[numbers[np.arange(num_to_extract)]] *= -1
    return initial_state


def eval_dynamics(
    hopfield: HopfieldNetwork, patterns: npt.NDArray[np.int64], args: argparse.Namespace
) -> None:
    """Evaluate dynamics of recall process in the network.

    Args:
        hopfield (HopfieldNetwork): The Hopfield network instance.
        patterns (npt.NDArray[np.int64]): The patterns to test.
        args (argparse.Namespace): The command line arguments.
    """
    index = np.random.choice(np.arange(args.num_patterns))
    initial_state = init_state(patterns[index], args.num_neurons, args.similarity)
    dircos_array = hopfield.recall(initial_state, patterns[index], args.num_steps)
    with open(args.log_file, "w", encoding="utf-8") as file_handler:
        for i in range(args.num_steps + 1):
            print(dircos_array[i], file=file_handler)


def main() -> None:
    """Run the Hopfield network simulation."""
    # 1. Set up argument parser
    args = parse_arguments()

    # 2. Create Hopfield network instance
    hopfield = HopfieldNetwork(args.num_neurons)

    # 3. Define bit patterns to be learned
    patterns = define_patterns(args.num_neurons, args.num_patterns)

    # 4. Learn patterns by Hebbian learning rule
    hopfield.learn(patterns)

    # 5. Evaluate dynamics of recall process
    eval_dynamics(hopfield, patterns, args)


if __name__ == "__main__":
    main()
