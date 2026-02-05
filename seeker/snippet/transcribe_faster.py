#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

#!/usr/bin/env python3
import sys
from pathlib import Path
from faster_whisper import WhisperModel

if len(sys.argv) < 2:
    print("Usage: python transcribe_faster.py <audio_file> [model] [language]")
    print("\nExamples:")
    print("  python transcribe_faster.py recording.m4a")
    print("  python transcribe_faster.py recording.m4a medium en")
    sys.exit(1)

input_file = sys.argv[1]
model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
language = sys.argv[3] if len(sys.argv) > 3 else "en"

# Initialize model
print(f"Loading {model_size} model...")
model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8",
)

# Transcribe
print(f"Transcribing {input_file}...")
segments, info = model.transcribe(
    input_file,
    language=language,
)

# Output file
output_file = Path(input_file).with_suffix('.txt')

# Write transcription
with open(output_file, 'w', encoding='utf-8') as f:
    for segment in segments:
        f.write(segment.text.strip() + '\n')

print(f"\nTranscription saved to: {output_file}")
print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
