#date: 2025-08-01T17:08:10Z
#url: https://api.github.com/gists/9289f83931b41fe7e6caeb109caeedb9
#owner: https://api.github.com/users/kordless

import zlib
from collections import deque
from typing import Deque

class RepetitionDetector:
    """
    Incrementally compresses data to detect repetition based on significant drops
    in the compression ratio relative to historical averages.

    This unified approach combines historical window tracking and relative threshold detection
    for robustness and efficiency.
    """

    def __init__(self, history_window: int = 5, detection_threshold: float = 0.2, sensitivity: float = 0.5):
        """
        Initializes the repetition detector.

        Args:
            history_window (int): Number of recent ratios stored for baseline calculation.
            detection_threshold (float): Absolute ratio threshold for early detection.
            sensitivity (float): Relative threshold to detect repetition against historical average.
        """
        self.compressor = zlib.compressobj()
        self.repetition_detected = False

        self.history_window: int = history_window
        self.detection_threshold: float = detection_threshold
        self.sensitivity: float = sensitivity

        self.ratio_history: Deque[float] = deque(maxlen=history_window)
        self.accumulated_data: bytes = b""
        self.chunks_processed: int = 0

    def add_chunk(self, chunk: bytes) -> bool:
        """
        Adds a chunk of data, compresses it, and checks for repetition.

        Args:
            chunk (bytes): Data chunk to process.

        Returns:
            bool: True if repetition is detected, False otherwise.
        """
        if self.repetition_detected or not chunk:
            return True

        self.accumulated_data += chunk
        self.chunks_processed += 1

        compressed_data = self.compressor.compress(chunk)
        compressed_size = len(compressed_data)
        original_size = len(chunk)

        current_ratio = compressed_size / original_size if original_size > 0 else 1.0

        if len(self.ratio_history) >= self.history_window:
            avg_historical_ratio = sum(self.ratio_history) / len(self.ratio_history)
            min_historical_ratio = min(self.ratio_history)

            if (current_ratio < avg_historical_ratio * self.sensitivity) or \
               (current_ratio < min_historical_ratio * self.detection_threshold):
                self.repetition_detected = True

        self.ratio_history.append(current_ratio)

        return self.repetition_detected

    def flush(self) -> bytes:
        """
        Flushes remaining data from the compressor.

        Returns:
            bytes: Final compressed data buffer.
        """
        return self.compressor.flush()

# Example usage
if __name__ == "__main__":
    detector = RepetitionDetector(history_window=5, detection_threshold=0.2, sensitivity=0.5)

    chunks = [
        b"This is the first sentence for our test data.",
        b"It contains a variety of words and structures.",
        b"The purpose is to establish a normal compression ratio.",
        b"Zlib will find some patterns but not extreme ones.",
        b"This fifth chunk completes the history window.",
        b"xyz_pattern_" * 20,  # Highly repetitive chunk
        b"xyz_pattern_" * 20,
        b"xyz_pattern_" * 20
    ]

    for idx, chunk in enumerate(chunks, 1):
        if detector.add_chunk(chunk):
            print(f"Repetition detected at chunk {idx}!")
            break
        else:
            print(f"Chunk {idx} processed without detecting repetition.")

    final_data = detector.flush()
    print(f"Final compressed data length: {len(final_data)} bytes.")
