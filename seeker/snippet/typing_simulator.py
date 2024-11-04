#date: 2024-11-04T16:56:59Z
#url: https://api.github.com/gists/9c5d109e7bdfce163e4315a10c199f5d
#owner: https://api.github.com/users/tobwen

import argparse
import random
import sys
import time
from typing import List, Tuple
import datetime

# Keyboard layout mapping for common typos (QWERTY layout)
ADJACENT_KEYS = {
    'a': ['q', 'w', 's', 'z'],
    'b': ['v', 'n', 'h', 'g'],
    'c': ['x', 'v', 'f', 'd'],
    'd': ['s', 'f', 'g', 'e', 'r'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'g', 'v', 'c', 'r', 't'],
    'g': ['f', 'h', 'b', 'v', 't', 'y'],
    'h': ['g', 'j', 'n', 'b', 'y', 'u'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'k', 'm', 'n', 'u', 'i'],
    'k': ['j', 'l', 'm', 'i', 'o'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'd', 'x', 'w'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'b', 'f', 'g'],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 'c', 's', 'd'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
    ' ': ['b', 'n', 'm']
}

class HumanTypingSimulator:
    def __init__(self,
                 typo_probability: float = 0.05,
                 correction_probability: float = 0.9,
                 delay_range: Tuple[float, float] = (0.05, 0.3)):
        self.typo_probability = typo_probability
        self.correction_probability = correction_probability
        self.delay_min, self.delay_max = delay_range

    def reseed_random(self):
        """Reseed random with current timestamp and microseconds"""
        now = datetime.datetime.now()
        seed = int(now.timestamp() * 1000000 + now.microsecond)
        random.seed(seed)

    def get_typing_delay(self, char: str) -> float:
        """Generate a human-like typing delay with context-based variability"""
        self.reseed_random()

        # Base delay using Gaussian distribution
        base_delay = random.gauss(
            (self.delay_min + self.delay_max) / 2,
            (self.delay_max - self.delay_min) / 6
        )

        # Add jitter for each character
        jitter = random.uniform(-0.02, 0.02)

        # Adjust delay based on character type
        if char.isspace():
            # Longer delay after spaces to simulate thinking between words
            pause = random.uniform(0.3, 0.6)
        elif char in ",.!?":
            # Slightly longer pause after punctuation marks
            pause = random.uniform(0.2, 0.4)
        else:
            # Regular character with jitter
            pause = 0

        final_delay = base_delay + jitter + pause
        return max(self.delay_min, min(final_delay, self.delay_max * 2))  # Allow for longer pauses

    def get_typo(self, char: str) -> str:
        """Get a possible typo for the given character"""
        self.reseed_random()
        char = char.lower()
        if char in ADJACENT_KEYS:
            return random.choice(ADJACENT_KEYS[char])
        return char

    def should_make_typo(self) -> bool:
        """Decide if we should make a typo"""
        self.reseed_random()
        return random.random() < self.typo_probability

    def should_correct_typo(self) -> bool:
        """Decide if we should correct a typo"""
        self.reseed_random()
        return random.random() < self.correction_probability

    def simulate_typing(self, text: str) -> None:
        """Simulate human typing with varied delays and contextual pauses"""
        buffer: List[str] = []

        for char in text:
            # Simulate typing delay
            time.sleep(self.get_typing_delay(char))

            # Decide if we make a typo
            if self.should_make_typo():
                typo = self.get_typo(char)
                buffer.append(typo)
                sys.stdout.write(typo)
                sys.stdout.flush()

                # Decide if we correct the typo
                if self.should_correct_typo():
                    time.sleep(self.get_typing_delay(char))
                    sys.stdout.write('\b \b')  # Backspace
                    sys.stdout.flush()
                    buffer.pop()

                    # Type the correct character
                    time.sleep(self.get_typing_delay(char))
                    buffer.append(char)
                    sys.stdout.write(char)
                    sys.stdout.flush()
            else:
                buffer.append(char)
                sys.stdout.write(char)
                sys.stdout.flush()

        # Check if the final buffer matches the input text
        final_text = ''.join(buffer)
        if final_text != text:
            # Clear the line and retype correctly
            sys.stdout.write('\r' + ' ' * len(buffer) + '\r')  # Clear the line
            sys.stdout.flush()
            for char in text:
                time.sleep(self.get_typing_delay(char))
                sys.stdout.write(char)
                sys.stdout.flush()

def parse_delay_range(delay_range_str: str) -> Tuple[float, float]:
    """Parse delay range string in format 'min-max'"""
    try:
        min_delay, max_delay = map(float, delay_range_str.split('-'))
        if min_delay < 0 or max_delay < 0 or min_delay >= max_delay:
            raise ValueError
        return (min_delay, max_delay)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Delay range must be in format 'min-max' where min and max are positive numbers and min < max"
        )

def main():
    parser = argparse.ArgumentParser(description='Simulate human-like typing with typos and delays')
    parser.add_argument('text', help='Text to type')
    parser.add_argument('--typo-probability', type=float, default=0.05,
                        help='Probability of making a typo (default: 0.05)')
    parser.add_argument('--correction-probability', type=float, default=0.9,
                        help='Probability of correcting a typo (default: 0.9)')
    parser.add_argument('--delay-range', type=parse_delay_range, default='0.05-0.3',
                        help='Delay range in seconds (format: min-max, default: 0.05-0.3)')

    args = parser.parse_args()

    simulator = HumanTypingSimulator(
        typo_probability=args.typo_probability,
        correction_probability=args.correction_probability,
        delay_range=args.delay_range
    )

    simulator.simulate_typing(args.text)
    sys.stdout.write('\n')

if __name__ == '__main__':
    main()

"""
# Default usage
python3 typing_simulator.py "Hello, World!"

# Fast typing (50-150ms between keystrokes)
python3 typing_simulator.py --delay-range 0.05-0.15 "Hello, World!"

# Very fast typing (20-80ms between keystrokes)
python3 typing_simulator.py --delay-range 0.02-0.08 "Hello, World!"

# Slow typing (200-500ms between keystrokes)
python3 typing_simulator.py --delay-range 0.2-0.5 "Hello, World!"

# Combined with other parameters
python3 typing_simulator.py --delay-range 0.02-0.08 --typo-probability 0.1 "Hello, World!"
"""
