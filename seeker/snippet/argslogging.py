#date: 2025-10-24T17:11:13Z
#url: https://api.github.com/gists/bb492ec7ae06e6e4e7f09d3a0b98ee06
#owner: https://api.github.com/users/4Falcon4

import argparse
import logging

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', nargs='?', type=argparse.FileType('r'),
                    default='data.txt', help='Specify input file')
parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'),
                    default='out.txt', help='Specify output file')
parser.add_argument('-l', '--logging', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set logging level')

args = parser.parse_args()

# Logging setup
logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')