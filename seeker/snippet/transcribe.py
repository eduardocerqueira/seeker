#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

#!/usr/bin/env python3
import argparse
from pathlib import Path
from faster_whisper import WhisperModel

def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using Faster-Whisper')
    parser.add_argument(
        'input',
        type=str,
        help='Input audio file',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'],
        help='Model size (default: base)',
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Language code (default: en)',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file (default: input.txt)',
    )
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading {args.model} model...")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )
    
    # Transcribe
    print(f"Transcribing {args.input}...")
    segments, info = model.transcribe(
        args.input,
        language=args.language,
    )
    
    # Determine output file
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.input).with_suffix('.txt')
    
    # Write transcription
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(segment.text.strip() + '\n')
    
    print(f"\nTranscription saved to: {output_path}")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

if __name__ == '__main__':
    main()
