#date: 2024-12-30T16:34:47Z
#url: https://api.github.com/gists/46ad52b412ac2c17651fd79a5dced3a3
#owner: https://api.github.com/users/ianhundere

#!/usr/bin/env python
"""
mlre sample converter
Converts audio samples to mlre-compatible mono format

examples:
    # default/basic usage (48kHz/24-bit wav)
    python convert_samps2mono.py

    # specific dir w/ custom settings
    python convert_samps2mono.py -i /path/to/samples -o mlre_samples -r 48000 -b 24 -f

    # aiff conversion
    python convert_samps2mono.py -e aiff

    # multiple formats in the same dir / used multiple times
    python convert_samps2mono.py -e aiff -o mlre_samples
    python convert_samps2mono.py -e wav -o mlre_samples -f
"""

import os
import sys
from argparse import ArgumentParser
from shutil import copyfile
import sox

# default mlre settings
DEFAULT_RATE = 48000
DEFAULT_BITS = 24
DEFAULT_EXT = 'wav'
OUTPUT_DIR = 'mlre_samples'

def parse_args():
    parser = ArgumentParser(description='Convert samples to mlre-compatible mono format')
    parser.add_argument('-i', '--input', type=str, default='.',
                       help='Input directory containing samples')
    parser.add_argument('-o', '--output', type=str, default=OUTPUT_DIR,
                       help='Output directory for converted samples')
    parser.add_argument('-e', '--ext', type=str, default=DEFAULT_EXT,
                       help='Input file extension to process')
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_RATE,
                       help='Target sample rate (default: 48000)')
    parser.add_argument('-b', '--bits', type=int, default=DEFAULT_BITS,
                       help='Target bit depth (default: 24)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Force overwrite of existing output directory')
    return parser.parse_args()

def setup_transformer(rate, bits):
    tfm = sox.Transformer()
    tfm.set_output_format(file_type='wav', rate=rate, bits=bits, channels=1)
    # subtle normalization to prevent clipping
    tfm.norm()
    return tfm

def convert_file(transformer, input_file, output_file):
    try:
        transformer.build(input_file, output_file)
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False

def main():
    args = parse_args()
    
    if os.path.exists(args.output) and not args.force:
        print(f'Error: Output directory "{args.output}" already exists.')
        print('Use -f/--force to overwrite')
        sys.exit(1)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    transformer = setup_transformer(args.rate, args.bits)
    input_ext = f'.{args.ext}'
    
    converted = 0
    errors = 0

    for root, _, files in os.walk(args.input):
        if root.startswith(os.path.join('.', args.output)):
            continue

        for file in files:
            if file.lower().endswith(input_ext):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, args.input)
                output_dir = os.path.join(args.output, rel_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.wav")

                needs_conversion = (
                    sox.file_info.channels(input_path) != 1 or
                    sox.file_info.sample_rate(input_path) != args.rate or
                    sox.file_info.bitdepth(input_path) != args.bits
                )

                if needs_conversion:
                    if convert_file(transformer, input_path, output_file):
                        converted += 1
                    else:
                        errors += 1
                else:
                    copyfile(input_path, output_file)
                    converted += 1

    print(f"\nConversion complete:")
    print(f"Successfully converted: {converted} files")
    if errors > 0:
        print(f"Errors encountered: {errors} files")

if __name__ == '__main__':
    main()
